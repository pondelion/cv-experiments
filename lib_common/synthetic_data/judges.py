"""画像の判定モデル(選択肢ごとの確率を返すモデル)で、画像に質問して答えの確率を得る(合成データの検品用。③ 組み合わせ)。

どの判定モデルも同じ使い方にそろえる:
    judge.ask(images, instruction, options) -> [{選択肢の値: 確率}, ...](画像ごと)
    options は {値: 説明} の順序付きの辞書。はい / いいえの質問は YES_NO を使う

- InternDecisionJudge: Intern-Decision(InternLM の decision model)。公式の推論のコード(モデルに付いている inference.py)の
  入力の組み立て(compile_row)をそのまま使い、複数の画像をまとめて 1 回で計算する(公式の DecisionEngine.predict は 1 件ずつ)。
  答えの記号の位置の確率を、モデルごとに決められた温度で較正する(公式と同じ: softmax(logits / 温度))
- OllamaVlmJudge: ollama の画像対応のモデル。選択肢に記号を付けて記号 1 文字で答えさせ、その 1 文字目の確率(logprobs)を
  記号の間で正規化する(ollama はまとめての処理ができないので 1 件ずつ)
"""
import base64
import io
import json
import sys
import urllib.request

import torch
from huggingface_hub import snapshot_download
from PIL import Image

YES_NO = {"no": "The answer is no.", "yes": "The answer is yes."}

# 合成データの 2 つのフィルターの質問(exp00002 で閾値を決めたときと同じ文面。変えたら閾値を決め直す)
#   (1) 物体のクラス: ollama qwen3.5:9b、頼んだクラスの確率が CLASS_KEEP_THRESHOLD 以上なら残す
#   (2) 背景の物体: Intern-Decision-0.8B、置いた物体を灰色の四角で隠した画像で「ある」の確率が BG_DROP_THRESHOLD より大きければ捨てる
CLASS_KEEP_THRESHOLD = 0.3
BG_DROP_THRESHOLD = 0.388  # 物体のある背景の見逃しが 1% になる閾値(exp00002)


def class_question(names: list[str]) -> tuple[str, dict[str, str]]:
    """(1) 物体のクラスの質問と選択肢(names + none)。

    Args:
        names: クラス名の並び。

    Returns:
        (質問の文, {選択肢の値: 選択肢の説明})。
    """
    return ("Which kind of object is the main object in the image?",
            {**{n: f"a {n}" for n in names}, "none": "none of these, or unclear"})


def background_question(names: list[str]) -> tuple[str, dict[str, str]]:
    """(2) 背景の物体の質問と選択肢(はい / いいえ)。

    Args:
        names: クラス名の並び。

    Returns:
        (質問の文, {"no": ..., "yes": ...})。
    """
    return ("Does the image contain at least one of these objects: " + ", ".join(names) +
            "? Plain gray rectangles are blank areas, not objects.", YES_NO)
MAX_SIDE = 512  # 判定モデルに渡す画像の長い辺(画像の分の計算を抑えるため)
STATE = "Look at the attached image."


def _resize(image: Image.Image) -> Image.Image:
    """長い辺が MAX_SIDE より大きければ、MAX_SIDE に縮める(縦横の比は変えない)。

    Args:
        image: 画像。

    Returns:
        縮めた(または元のままの)画像。
    """
    s = MAX_SIDE / max(image.size)
    return image.resize((round(image.width * s), round(image.height * s))) if s < 1 else image


class InternDecisionJudge:
    """Intern-Decision で、画像ごとに選択肢の確率を出す。

    Args:
        repo: Hugging Face のリポジトリ名(キャッシュから読む)。
        batch_size: まとめて計算する画像の数(以前の検証で 16 前後が速かった)。
        device: 計算する装置。
    """

    def __init__(self, repo: str = "internlm/Intern-Decision-0.8B", batch_size: int = 16, device: str = "cuda"):
        # モデルに付いている inference.py はリンクの実体の場所(blobs/)からモデルを探してしまうので、場所を明示して渡す
        self.checkpoint = snapshot_download(repo, local_files_only=True)
        sys.path.insert(0, self.checkpoint)
        import inference as inf  # モデルに付いている公式の推論のコード(中身は確認済み: ネットワーク・外部コマンドなし)
        self.inf = inf
        engine = inf.DecisionEngine(checkpoint=self.checkpoint, device=device)
        self.processor, self.tokenizer, self.model = engine.backend.processor, engine.backend.tokenizer, engine.backend.model
        self.marker_id, self.temperature = engine.backend.marker_id, engine.temperature
        self.batch_size, self.device = batch_size, device

    def _render(self, instruction: str, options: dict[str, str]):
        """公式の compile_row で、画像 1 枚の質問の文を組み立てる(画像の部分は processor が埋める)。

        Args:
            instruction: 質問の文。
            options: {選択肢の値: 選択肢の説明}。

        Returns:
            (モデルに入れる文, 選択肢の記号のトークンの ID の並び)。
        """
        row = {"state": STATE, "questions": {"answer": {"type": "choice", "instructions": instruction, "criteria": options}},
               "images": ["image"]}
        compiled = self.inf.compile_row(row)
        messages = compiled.messages
        messages[1]["content"] = [{"type": "image"} if p["type"] == "image_url" else p for p in messages[1]["content"]]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                  enable_thinking=False, add_vision_id=True)
        symbol_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in compiled.symbols["answer"]]
        return text, symbol_ids

    @torch.inference_mode()
    def ask(self, images: list[Image.Image], instruction: str, options: dict[str, str]) -> list[dict[str, float]]:
        """画像ごとに、同じ質問の選択肢の確率を出す(batch_size 枚ずつまとめて計算する)。

        Args:
            images: 画像の並び。
            instruction: 質問の文。
            options: {選択肢の値: 選択肢の説明}。

        Returns:
            画像ごとの {選択肢の値: 確率}(確率の和は 1)。
        """
        text, symbol_ids = self._render(instruction, options)
        values = list(options)
        out = []
        self.processor.tokenizer.padding_side = "right"
        for i in range(0, len(images), self.batch_size):
            chunk = [_resize(im.convert("RGB")) for im in images[i: i + self.batch_size]]
            batch = self.processor(text=[text] * len(chunk), images=chunk, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**batch, use_cache=False).logits  # (画像の数, 文のトークンの数, 語彙の数)
            for b in range(len(chunk)):
                pos = int((batch["input_ids"][b] == self.marker_id).nonzero()[0]) - 1  # 答えの記号を予測する位置
                # その位置の、選択肢の記号の値だけを取って確率にする: (語彙の数,) -> (選択肢の数,)
                probs = torch.softmax(logits[b, pos, symbol_ids].float() / self.temperature, dim=-1).tolist()
                out.append(dict(zip(values, probs)))
        return out


class OllamaVlmJudge:
    """ollama の画像対応のモデルで、画像ごとに選択肢の確率を出す(答えの記号 1 文字目の確率を正規化する)。

    Args:
        model: ollama のモデル名。
        timeout: 1 回の依頼の待ち時間の上限(秒)。
    """

    URL = "http://localhost:11434/api/generate"
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, model: str = "qwen3.5:9b", timeout: float = 300):
        self.model, self.timeout = model, timeout

    def ask(self, images: list[Image.Image], instruction: str, options: dict[str, str]) -> list[dict[str, float]]:
        """画像ごとに、同じ質問の選択肢の確率を出す(1 枚ずつ依頼する)。

        Args:
            images: 画像の並び。
            instruction: 質問の文。
            options: {選択肢の値: 選択肢の説明}。

        Returns:
            画像ごとの {選択肢の値: 確率}(確率の和は 1)。
        """
        values = list(options)
        letters = self.LETTERS[: len(values)]
        listing = "\n".join(f"{l}. {v}: {d}" for l, (v, d) in zip(letters, options.items()))
        prompt = f"{instruction}\nAnswer with only the letter of one option.\n{listing}\nAnswer:"
        out = []
        for im in images:
            buf = io.BytesIO()
            _resize(im.convert("RGB")).save(buf, format="JPEG", quality=90)
            req = {"model": self.model, "stream": False, "think": False, "logprobs": True, "top_logprobs": 20,
                   "options": {"temperature": 0, "num_predict": 1}, "prompt": prompt,
                   "images": [base64.b64encode(buf.getvalue()).decode()]}
            with urllib.request.urlopen(urllib.request.Request(self.URL, json.dumps(req).encode(),
                                                               {"Content-Type": "application/json"}), timeout=self.timeout) as r:
                lp = json.load(r)["logprobs"][0]["top_logprobs"]
            score = {}
            for t in lp:  # " A" と "A" のような同じ記号の候補は確率を足す
                key = t["token"].strip()
                if key in letters:
                    score[key] = score.get(key, 0.0) + float(torch.tensor(t["logprob"]).exp())
            total = sum(score.values())
            out.append({v: (score.get(l, 0.0) / total if total else 1 / len(values)) for l, v in zip(letters, values)})
        return out

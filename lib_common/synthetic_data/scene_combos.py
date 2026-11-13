"""合成の場面に置くクラスの組み合わせ(場所 + 一緒に写ってよくあるクラス)を、手元の ollama の LLM で事前に作る。

場所と合わない組み合わせ(牛と飛行機を台所に置くなど)では、背景の inpainting がうまく描けず、場面としても不自然になるので、
生成のときは、この組み合わせのプールの中から選ぶ。作り方は 3 段:

1. 場所: クラスごとに、そのクラスが写っている写真としてよくある場所を PLACES の中から選ばせる
2. 組み合わせ: (a) 場所ごとに、1. でその場所を選んだクラスの中から、一緒に写っていてよくある 2〜3 クラスの組み合わせを挙げさせる。
   (b) クラスごとに、そのクラスとよく一緒に写る 1〜2 クラスと場所を挙げさせる((a) で組み合わせの少ないクラスも拾うため)
3. 確認: 2. の組み合わせを 1 つずつ、その場所の 1 枚の写真に一緒に写っているのがどれだけよくあるかを 1〜5 で答えさせ、
   MIN_RATING 以上のものだけ残す(2. で挙げた組み合わせの誤りを落とす。点は全部記録するので、閾値を変えて選び直せる)

LLM は gpt-oss:20b に考えさせて(think)使う。qwen3.5:9b に考えさせずに即答させると、場所の選び方が甘く(猫が空に、テレビが
高速道路に写るなど)、組み合わせの確認も自然な組み合わせ(居間のソファと人)を落とし不自然な組み合わせ(港の飛行機と椅子)を残した。
qwen3.5:9b に考えさせると同じくらい良くなるが、1 回 30 秒ほどかかる(gpt-oss:20b は数秒)。

場所を決まった一覧から選ばせるのは、背景の文面を場所ごとに 1 回だけ特徴に変えて使い回すため。
"""
import json
import re
import urllib.request

from lib_common.synthetic_data.llm_prompts import OLLAMA_URL

MODEL = "gpt-oss:20b"
THINK = "medium"  # gpt-oss の考える量(low / medium / high)

# 場所の一覧(背景の文面 "a photo of an empty {場所}" になる)。prompts.CONTEXTS の場所に、よくある場所を足したもの
PLACES = ["city street", "park", "living room", "parking lot", "airport runway", "sky", "railway station", "harbor",
          "lake", "garden", "farm field", "kitchen", "dining room", "beach", "country road", "highway", "river",
          "backyard", "bedroom", "office", "restaurant", "cafe", "campsite", "bus station", "mountain trail"]
MIN_RATING = 3  # 3 = ありうるがよくはない。4 以上だけにすると、自然な組み合わせ(駐車場のバス・車・バイクなど)も落ちた

PLACES_PROMPT = ("Where is a {name} commonly seen in everyday photos? From this list, choose only the places where a photo "
                 "showing a {name} is common and typical (not merely possible): {places}.\n"
                 'Answer with only JSON: {{"places": ["...", ...]}}')
COMBOS_PROMPT = ("Objects that are commonly seen in a {place}: {names}.\n"
                 "List up to 15 of the most common combinations of 2 or 3 of these objects that appear together in one "
                 "everyday photo of a {place} (typical scenes, not merely possible ones). "
                 "Use the object names exactly as written.\n"
                 'Answer with only JSON: {{"combinations": [["...", "..."], ["...", "...", "..."], ...]}}')
PARTNERS_PROMPT = ("Objects: {names}.\nPlaces: {places}.\n"
                   "List up to 10 typical everyday photos in which a {name} appears together with 1 or 2 other objects "
                   "from the list. "
                   "For each, give the other objects and the place (from the list of places). Use the names exactly as written.\n"
                   'Answer with only JSON: {{"scenes": [{{"objects": ["..."], "place": "..."}}, ...]}}')
RATE_PROMPT = ("How common is an everyday photo of a {place} that shows {objects} together?\n"
               "1 = unrealistic, 2 = very unusual, 3 = possible but uncommon, 4 = common, 5 = very common.\n"
               'Answer with only JSON: {{"rating": <1-5>}}')


PLACE_DESC_PROMPT = ("Write {n} diverse, short visual descriptions of an empty {place} for photorealistic image generation. "
                     "Vary the time of day, weather and lighting, season, architectural style or era, region of the world, "
                     "and camera viewpoint. Each must describe only the place itself. "
                     "Do not mention or imply any of these objects: {names}.\n"
                     'Answer with only JSON: {{"descriptions": ["...", ...]}}')
DESC_PER_CALL = 16  # 1 回の依頼で作らせる説明文の数(多いと考える部分が長くなり、答えが打ち切られやすい)
# 20 クラスとその言い換え。説明文にこれらの言葉があると背景に描かれやすいので、その説明文は使わない
# (LLM に書かないよう頼んでも、港の説明文に "a single fishing vessel" などが入った)
CLASS_WORDS = re.compile(
    r"\b(person|people|man|woman|child|crowd|passenger|bicycle|bike|bird|gull|boat|vessel|ship|yacht|sailboat|ferry|canoe|"
    r"kayak|bottle|bus|car|cars|vehicle|truck|taxi|cat|chair|stool|cow|cattle|table|dog|horse|motorcycle|scooter|plant|"
    r"potted|sheep|couch|sofa|train|locomotive|tram|tv|television|screen|monitor)s?\b", re.I)


def place_descriptions(place: str, n: int, names: list[str], model: str = MODEL) -> list[str]:
    """場所の背景の、多様な説明文を作る(同じ場所の背景が似た画像ばかりになるのを防ぐため)。

    固定の文面("a photo of an empty {場所}")だと、乱数の種を変えても似た背景になった(exp00202: 同じ場所の背景どうしの
    特徴のコサイン類似度 0.825。16 個の説明文に変えると 0.471)。DESC_PER_CALL 個ずつ、乱数の種を変えて n 個まで作らせる。

    Args:
        place: 場所。
        n: 作る説明文の数。
        names: 説明文に書かせないクラス名(背景に描かれると、正解のない物体になる)。
        model: ollama のモデル名。

    Returns:
        説明文のリスト(長さ n 以下。CLASS_WORDS の言葉を含むもの・重なったものは除く)。
    """
    out: list[str] = []
    for r in range(3 * -(-n // DESC_PER_CALL)):  # 除いた分を補うため、必要な回数の 3 倍まで
        got = ask_json(PLACE_DESC_PROMPT.format(n=DESC_PER_CALL, place=place, names=", ".join(names)), "descriptions",
                       model, seed=1000 * r, temperature=0.0 if r == 0 else 0.8) or []
        out += [d.strip() for d in got if isinstance(d, str) and d.strip() and not CLASS_WORDS.search(d) and d.strip() not in out]
        if len(out) >= n:
            break
    return out[:n]


def ask_json(prompt: str, key: str, model: str = MODEL, think=THINK, tries: int = 5, seed: int = 0,
             temperature: float = 0.0):
    """考えさせてから JSON で答えさせ、答えの JSON の key の値を返す(形が崩れた・空の答えは、温度を上げて聞き直す)。

    ollama の JSON の形式の指定(format)は考える設定と一緒に使うと答えが空になったので、指定せず答えの文から取り出す。
    考える部分が長いと答えが出る前に出力の長さの上限で打ち切られて空になる(1 回目の作成で 8 か所の組み合わせが 0 個になった。
    居間で確かめると、考える部分で組み合わせを全部数え上げて 8,192 トークンで打ち切られていた)ので、
    問いで挙げる数を絞り、出力の長さの上限を大きくし、空なら聞き直す。

    Args:
        prompt: 問いの文。
        key: 答えの JSON から取り出すキー。
        model: ollama のモデル名。
        think: 考える量(gpt-oss は "low" / "medium" / "high"、ほかのモデルは True / False)。
        tries: 聞き直す回数の上限。
        seed: LLM の乱数の種(聞き直すたびに 1 ずつ足す)。
        temperature: 1 回目の温度(聞き直すときは 0.7 以上にする)。

    Returns:
        key の値(tries 回とも取れなければ None)。
    """
    for t in range(tries):
        req = {"model": model, "stream": False, "think": think, "prompt": prompt,
               "options": {"temperature": temperature if t == 0 else max(temperature, 0.7), "seed": seed + t,
                           "num_predict": 16384, "num_ctx": 20480}}
        with urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, json.dumps(req).encode(),
                                                           {"Content-Type": "application/json"}), timeout=900) as r:
            text = json.load(r)["response"]
        m = re.search(r"\{.*\}", text, re.S)
        try:
            value = json.loads(m.group()).get(key) if m else None
        except json.JSONDecodeError:
            value = None
        if value:
            return value
    return None


def _join(names: list[str]) -> str:
    """クラス名の並びを、冠詞を付けた英語の並びの文にする(例: ["person", "dog", "car"] → "a person, a dog and a car")。

    Args:
        names: クラス名の並び(2 個以上)。

    Returns:
        英語の文。
    """
    items = [("an " if n[0] in "aeiou" else "a ") + n for n in names]
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_pool(names: list[str], model: str = MODEL, log=print) -> dict:
    """組み合わせのプールを作る。

    Args:
        names: クラス名の並び。
        model: ollama のモデル名。
        log: 途中経過を書く関数。

    Returns:
        {"places": {クラス: [場所, ...]}, "combos": [{"place": 場所, "classes": [クラス, ...], "rating": 点}, ...]
         (MIN_RATING 以上だけ), "all_combos": 確認した全部, "min_rating": MIN_RATING}。
    """
    places = {}
    for n in names:
        got = ask_json(PLACES_PROMPT.format(name=n, places=", ".join(PLACES)), "places", model) or []
        places[n] = [p for p in PLACES if p in got]  # 一覧にない場所は捨てる
        log(f"場所 {n}: {places[n]}")
    candidates = set()

    def add(place: str, objs: list, allowed: list[str]) -> None:
        """LLM が挙げた組み合わせを、候補に足す(一覧にない場所・クラス、重なり、2〜3 クラスでないものは捨てる)。

        Args:
            place: 場所。
            objs: クラス名の並び。
            allowed: 使ってよいクラス名。
        """
        c = sorted({x for x in objs if x in allowed})
        if place in PLACES and 2 <= len(c) <= 3:
            candidates.add((place, tuple(c)))

    # 2a. 場所ごとに、その場所によく写るクラスの組み合わせ
    for place in PLACES:
        here = [n for n in names if place in places[n]]
        if len(here) < 2:
            continue
        for c in ask_json(COMBOS_PROMPT.format(place=place, names=", ".join(here)), "combinations", model) or []:
            if isinstance(c, list):
                add(place, c, here)
        log(f"組み合わせ(場所ごと){place}: {sum(p == place for p, _ in candidates)} 個")
    # 2b. クラスごとに、そのクラスとよく一緒に写るクラスと場所(2a で組み合わせの少ないクラスも拾う)
    for n in names:
        before = len(candidates)
        for s in ask_json(PARTNERS_PROMPT.format(name=n, names=", ".join(names), places=", ".join(PLACES)),
                          "scenes", model) or []:
            if isinstance(s, dict) and isinstance(s.get("objects"), list):
                add(s.get("place"), [n] + s["objects"], names)
        log(f"組み合わせ(クラスごと){n}: {len(candidates) - before} 個増えた")
    all_combos = []
    for place, c in sorted(candidates):
        rating = ask_json(RATE_PROMPT.format(objects=_join(list(c)), place=place), "rating", model)
        all_combos.append({"place": place, "classes": list(c), "rating": rating if isinstance(rating, int) else 0})
    kept = [c for c in all_combos if c["rating"] >= MIN_RATING]
    log(f"確認: {len(all_combos)} 個中 {len(kept)} 個を残した")
    return {"places": places, "combos": kept, "all_combos": all_combos, "min_rating": MIN_RATING}


def unload(model: str = MODEL) -> None:
    """ollama のモデルを GPU から下ろす。

    Args:
        model: 下ろす ollama のモデル名。
    """
    req = {"model": model, "keep_alive": 0}
    urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}), timeout=60).read()

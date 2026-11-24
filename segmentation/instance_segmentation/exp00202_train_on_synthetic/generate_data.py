"""合成データ voc20_v1 を作って保存する(exp00202 以降で使う。① 生成だけ の作り方で作り、フィルターの確率は記録だけする)。

作り方は exp00201 で決めたもの:
    1 段目 objects:   クラスごとに N_DESC 個の説明文を ollama の LLM で作り(llm_v3: 向き・ポーズを割り当てる)、物体を 1 つずつ
                      無地の背景(クラスごとの色)で生成し、背景の色の強さの規則で抜き出す
    filter1:          (1) 物体のクラス: ollama qwen3.5:9b で「頼んだクラスの確率」を記録する(捨てない)
    2 段目 scenes:    場所を 1 つ選び、その場所にいそうなクラスの物体を 1〜4 個置く(小さめ・重なりを抑えた置き方)。背景は 2 通り:
                        paste:   場所だけの画像を生成して貼る
                        inpaint: 物体だけを灰色の無地に置き、物体以外を FLUX.2 [klein] の inpainting で一から描く(全クラスを否定する文面)
                      (2) 背景の物体: Intern-Decision-0.8B で「20 クラスのどれかがあるか」の確率を記録する(捨てない)。
                      paste は貼る前の背景の画像、inpaint は置いた物体を灰色の四角で隠した画像を判定させる
① のデータは全部の場面を使い、③ のデータはフィルターの閾値(lib_common.synthetic_data.judges)を通った場面だけを使う。
どの段も、作った分は飛ばして続きから再開できる。

出力(datasets/synthetic/voc20_v1/):
    objects/descriptions.json、objects/{番号}.jpg(生成した画像)、objects/{番号}_mask.png、objects/objects.jsonl
    scenes/{paste,inpaint}/{番号}.jpg、scenes/scenes.jsonl(場面ごとの物体・正解のマスク(RLE)・フィルターの確率)

使い方: .venv/bin/python -u segmentation/instance_segmentation/exp00202_train_on_synthetic/generate_data.py --stage all
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import common
import cv2
import numpy as np
import torch
from PIL import Image

from lib_common import DATASETS_DIR
from lib_common.classes import VOC20
from lib_common.masks import mask_to_rle
from lib_common.synthetic_data.chroma_key import is_usable, key_for_background
from lib_common.synthetic_data.compose import blend, compose_layers, crop_object, random_layout
from lib_common.synthetic_data.flux2_klein import (DEFAULT_STEPS, encode_prompts, inpaint_pipeline, load_pipeline,
                                                   offload_text_encoder)
from lib_common.synthetic_data.judges import (InternDecisionJudge, OllamaVlmJudge, background_question,
                                              class_question)
from lib_common.synthetic_data.llm_prompts import describe_v3, unload
from lib_common.synthetic_data.prompts import (background_for, background_prompt,
                                               object_prompt_from_description)
from lib_common.synthetic_data import scene_combos

OUT = DATASETS_DIR / "synthetic" / "voc20_v1"
NAMES = [n for n, _ in VOC20]
SIZE = 768
N_DESC = 150  # クラスごとの物体の数(フィルターで落ちる分を見込む)
DESC_CHUNK = 30  # LLM に 1 回で作らせる説明文の数
CELL = 16  # FLUX.2 [klein] の inpainting のマスクが縮められるマス目(exp00201 の inpaint_full と同じ)
JPEG_QUALITY = 90
MAX_OBJECTS = 4  # 1 つの場面に置く物体の数の上限(前の版の作り方と同じ。多いと重なりで置けない物体が増える)
COMBO_TOP = 5  # 組み合わせを選ぶとき、ほかのクラスの数が少ない上位いくつからランダムに選ぶか
N_PLACE_DESC = 32  # 場所ごとの背景の説明文の数
SCENE_CHUNK = 200  # 背景の文面をまとめて特徴に変える場面の数
# 背景の文面を工夫しても、判定モデルで 20 クラスがあるとされた割合が 40% 以上だった場所(exp00202 の README)。使わない
EXCLUDED_PLACES = ["cafe", "harbor", "restaurant"]


def read_jsonl(path) -> list[dict]:
    """JSON Lines のファイルを読む。

    Args:
        path: ファイルのパス(Path)。

    Returns:
        行ごとの dict の並び(ファイルがなければ空)。
    """
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage_descriptions() -> None:
    """1 段目の前: クラスごとに N_DESC 個の物体の説明文を LLM で作り、objects/descriptions.json に保存する(クラスごとに再開できる)。"""
    path = OUT / "objects" / "descriptions.json"
    desc = json.loads(path.read_text()) if path.exists() else {}
    for name in NAMES:
        if len(desc.get(name, [])) >= N_DESC:
            continue
        desc[name] = [d for s in range(N_DESC // DESC_CHUNK) for d in describe_v3(name, DESC_CHUNK, seed=s)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(desc, ensure_ascii=False, indent=1))  # クラスごとに保存(途中から再開できる)
        print("説明文", name, len(desc[name]), flush=True)


def stage_objects() -> None:
    """1 段目: 説明文ごとに物体を無地の背景で生成し、背景の色の規則で抜き出して保存する(作った物体は飛ばす)。"""
    desc = json.loads((OUT / "objects" / "descriptions.json").read_text())
    rec_path = OUT / "objects" / "objects.jsonl"
    done = {r["id"] for r in read_jsonl(rec_path)}
    pipe = load_pipeline()
    rng = random.Random(0)
    with open(rec_path, "a") as f:
        for ci, name in enumerate(NAMES):
            # rng は飛ばした分も同じ順に進める
            todo = [(ci * 1000 + j, d, object_prompt_from_description(d, name, rng, add_view=False))
                    for j, d in enumerate(desc[name][:N_DESC])]
            todo = [t for t in todo if t[0] not in done]
            if not todo:
                continue
            # クラスごとに文面を先にまとめて特徴に変え、文章を読む部分(約 8GB)を CPU に退かせてから生成する
            # (生成中の GPU のメモリを約 21GB → 約 11GB に減らし、並べて動かす学習の分を空ける。文面を変える間だけ約 18GB)
            pipe.text_encoder.to("cuda")
            embs = [e.cpu() for e in encode_prompts(pipe, [p for _, _, p in todo])]
            offload_text_encoder(pipe)
            for (oid, d, prompt), emb in zip(todo, embs):
                image = np.array(pipe(prompt_embeds=emb.cuda(), height=SIZE, width=SIZE, num_inference_steps=DEFAULT_STEPS,
                                      generator=torch.Generator("cuda").manual_seed(oid)).images[0])
                bg = background_for(name)
                mask, stats = key_for_background(image, bg)
                Image.fromarray(image).save(OUT / "objects" / f"{oid:05d}.jpg", quality=JPEG_QUALITY)
                Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / "objects" / f"{oid:05d}_mask.png")
                f.write(json.dumps({"id": oid, "class": name, "description": d, "prompt": prompt, "background": bg,
                                    "stats": stats, "usable": is_usable(stats)}) + "\n")
                f.flush()
            print("物体", name, flush=True)


def stage_filter1() -> None:
    """フィルター (1): 生成した物体ごとに、ollama の VLM で「頼んだクラスの確率」を記録する(捨てない。objects/filter1.json)。"""
    recs = read_jsonl(OUT / "objects" / "objects.jsonl")
    out_path = OUT / "objects" / "filter1.json"
    scores = json.loads(out_path.read_text()) if out_path.exists() else {}
    judge = OllamaVlmJudge()
    instruction, options = class_question(NAMES)
    for r in recs:
        if str(r["id"]) in scores:
            continue
        p = judge.ask([Image.open(OUT / "objects" / f"{r['id']:05d}.jpg")], instruction, options)[0]
        scores[str(r["id"])] = p[r["class"]]  # 頼んだクラスの確率
        if len(scores) % 200 == 0:
            out_path.write_text(json.dumps(scores))
            print("filter1", len(scores), flush=True)
    out_path.write_text(json.dumps(scores))


def load_library() -> dict[str, list]:
    """規則で使えると判定した物体を、クラスごとの切り抜きの一覧にする(フィルターの確率に関わらず全部。① 用)。

    Returns:
        {クラス名: [切り抜き]}。
    """
    lib: dict[str, list] = {}
    for r in read_jsonl(OUT / "objects" / "objects.jsonl"):
        if r["usable"]:
            img = np.array(Image.open(OUT / "objects" / f"{r['id']:05d}.jpg").convert("RGB"))
            mask = np.array(Image.open(OUT / "objects" / f"{r['id']:05d}_mask.png")) > 127
            obj = crop_object(r["class"], img, mask, r["stats"]["bg_color"])
            obj.id = r["id"]
            lib.setdefault(r["class"], []).append(obj)
    return lib


def gray_boxes(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """置いた物体を、枠を 10% + 4 画素広げた灰色の四角で隠す(exp00002 と同じ隠し方)。

    Args:
        image: 画像 (H, W, 3)。
        masks: 置いた物体のマスク (H, W) の並び。

    Returns:
        隠した画像 (H, W, 3)。
    """
    out = image.copy()
    for m in masks:
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        dx, dy = (x1 - x0) * 0.1 + 4, (y1 - y0) * 0.1 + 4
        out[max(0, int(y0 - dy)): int(y1 + dy) + 1, max(0, int(x0 - dx)): int(x1 + dx) + 1] = 128
    return out


def stage_combos() -> None:
    """場所とクラスの組み合わせのプールを LLM で作って保存する(あれば作らない)。"""
    path = OUT / "scenes" / "combos.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pool = scene_combos.build_pool(NAMES, log=lambda s: print(s, flush=True))
    scene_combos.unload()
    path.write_text(json.dumps(pool, indent=1))


def stage_place_descriptions() -> None:
    """場所ごとの背景の説明文を LLM で作って保存する(作った場所は飛ばす)。外す場所(EXCLUDED_PLACES)は作らない。"""
    pool = json.loads((OUT / "scenes" / "combos.json").read_text())
    path = OUT / "scenes" / "place_descriptions.json"
    desc = json.loads(path.read_text()) if path.exists() else {}
    places = sorted({p for ps in pool["places"].values() for p in ps} - set(EXCLUDED_PLACES))
    for p in places:
        if len(desc.get(p, [])) >= N_PLACE_DESC:
            continue
        desc[p] = scene_combos.place_descriptions(p, N_PLACE_DESC, NAMES)
        path.write_text(json.dumps(desc, ensure_ascii=False, indent=1))  # 場所ごとに保存(途中から再開できる)
        print("場所の説明文", p, len(desc[p]), flush=True)
    scene_combos.unload()


def plan_scenes(n: int, lib: dict[str, list], pool: dict, n_desc: dict[str, int],
                rng: random.Random) -> list[tuple[str, int, list]]:
    """n 個の場面(場所, 場所の説明文の番号, 物体の置き方)を、クラスごとの物体の数がそろうように先に決める。

    場所とクラスの組み合わせは、LLM で事前に作った組み合わせのプール(pool。lib_common.synthetic_data.scene_combos)から選ぶ
    (EXCLUDED_PLACES の場所は使わない)。背景の説明文は、その場所の n_desc[場所] 個からランダムに選ぶ。
    場面ごとに:
        1. 1 枚に出るクラスの数 k を 1〜3 から、物体の総数 m を k〜MAX_OBJECTS から、それぞれ等しい確率で選ぶ
           (単一クラスの場面と複数クラスの場面、1 クラスあたりの物体の数の両方に幅を持たせる)
        2. 1 つ目のクラス = これまでに置いた物体がいちばん少ないクラス(同じ数ならランダム)
        3. k = 1 なら、そのクラスがよく写る場所(pool["places"])をランダムに選ぶ。
           k ≥ 2 なら、そのクラスを含む k クラスの組み合わせ(pool["combos"])のうち、ほかのクラスの置いた物体の合計が
           少ない COMBO_TOP 個からランダムに選ぶ(数をそろえつつ、毎回同じ組み合わせにならないように)。k クラスの組み合わせが
           なければ k を 1 つずつ減らす
        4. m 個を k クラスに、どのクラスも 1 個以上になるようランダムに分ける
    数は場面の先頭から順にそろえていくので、先頭の何枚を使っても(2k・5k など)クラスごとの数はほぼそろう。
    数えるのは置き方を決めた物体(重なりで置けなかった物体は数えない)。

    Args:
        n: 場面の数。
        lib: load_library の返り値。
        pool: 場所とクラスの組み合わせのプール。
        n_desc: 場所ごとの背景の説明文の数。
        rng: 乱数。

    Returns:
        場面ごとの (場所, 説明文の番号, 物体の置き方)。
    """
    count = {x: 0 for x in lib}
    combos = [c for c in pool["combos"] if all(x in lib for x in c["classes"]) and c["place"] not in EXCLUDED_PLACES]
    places = {x: [p for p in ps if p not in EXCLUDED_PLACES] for x, ps in pool["places"].items()}
    assert all(places[x] for x in lib), "外した場所のせいで、置ける場所がなくなったクラスがある"
    plans = []
    for _ in range(n):
        k, low = rng.randint(1, 3), min(count.values())
        first = rng.choice([x for x in lib if count[x] == low])
        names, context = [first], rng.choice(places[first])
        for kk in range(k, 1, -1):
            cands = [c for c in combos if first in c["classes"] and len(c["classes"]) == kk]
            if cands:
                rng.shuffle(cands)  # 合計が同じ組み合わせの順をランダムに
                cands.sort(key=lambda c: sum(count[x] for x in c["classes"] if x != first))
                c = rng.choice(cands[:COMBO_TOP])
                names, context = [first] + [x for x in c["classes"] if x != first], c["place"]
                break
        m = rng.randint(len(names), MAX_OBJECTS)
        per = [1] * len(names)
        for _ in range(m - len(names)):
            per[rng.randrange(len(names))] += 1
        objects = [rng.choice(lib[x]) for x, c in zip(names, per) for _ in range(c)]
        rng.shuffle(objects)  # 置く順(重なったときに置けなくなる物体)がクラスで偏らないように
        layout = random_layout(objects, SIZE, rng)
        for p in layout:
            count[p.obj.name] += 1
        plans.append((context, rng.randrange(n_desc[context]), layout))
    return plans


def stage_scenes(n: int, out_dir: Path) -> None:
    """場面を作って out_dir に保存する(作った分は飛ばす)。

    背景の文面は、場所の説明文 + 全クラスを名前で否定する文(prompts.background_prompt)。paste の背景も inpaint も同じ文面。
    説明文の特徴は、SCENE_CHUNK 個の場面ごとに、その分だけまとめて変える(全部を先に変えると CPU のメモリが約 17GB 要る)。
    1 場面あたりの時間の内訳(背景・inpaint・判定・保存)を 100 場面ごとに表示する。

    Args:
        n: 場面の数。
        out_dir: 保存先(scenes/。動作確認では別のフォルダ)。
    """
    rec_path = out_dir / "scenes.jsonl"
    done = len(read_jsonl(rec_path))
    for v in ["paste", "inpaint"]:
        (out_dir / v).mkdir(parents=True, exist_ok=True)
    lib = load_library()
    pool = json.loads((OUT / "scenes" / "combos.json").read_text())
    desc = json.loads((OUT / "scenes" / "place_descriptions.json").read_text())
    # 場面は先に全部決める(再開しても同じ場面になる)
    plans = plan_scenes(n, lib, pool, {p: len(v) for p, v in desc.items()}, random.Random(0))
    pipe = load_pipeline()
    offload_text_encoder(pipe)
    inpaint = inpaint_pipeline(pipe)
    judge = InternDecisionJudge()
    bg_instruction, yes_no = background_question(NAMES)
    times = {"background": 0.0, "inpaint": 0.0, "judge": 0.0, "save": 0.0, "encode": 0.0}
    emb = {}
    with open(rec_path, "a") as f:
        for k in range(done, n):
            context, di, layout = plans[k]
            if (context, di) not in emb:  # この先 SCENE_CHUNK 場面の分の文面を、まとめて特徴に変える
                t = time.perf_counter()
                need = sorted({(c, d) for c, d, _ in plans[k: k + SCENE_CHUNK]})
                pipe.text_encoder.to("cuda")
                emb = {key: e.cpu() for key, e in zip(need, encode_prompts(
                    pipe, [background_prompt(desc[c][d], NAMES) for c, d in need]))}
                offload_text_encoder(pipe)
                times["encode"] += time.perf_counter() - t
            e = emb[(context, di)].cuda()
            t = time.perf_counter()
            gen = torch.Generator("cuda").manual_seed(k)
            bg = np.array(pipe(prompt_embeds=e, height=SIZE, width=SIZE,
                               num_inference_steps=DEFAULT_STEPS, generator=gen).images[0])
            times["background"] += time.perf_counter() - t
            t = time.perf_counter()
            fg, alpha, gts, kept = compose_layers((SIZE, SIZE), layout, return_kept=True)
            union = np.zeros((SIZE, SIZE), bool)
            for _, m in gts:
                union |= m
            keep = cv2.erode(union.astype(np.uint8), np.ones((CELL, CELL), np.uint8)).astype(bool)
            canvas = Image.fromarray(blend(np.full((SIZE, SIZE, 3), 128, np.uint8), fg, alpha))
            painted = np.array(inpaint(prompt_embeds=e, image=canvas,
                                       mask_image=Image.fromarray((~keep * 255).astype(np.uint8)), height=SIZE, width=SIZE,
                                       strength=1.0, guidance_scale=1.0, num_inference_steps=DEFAULT_STEPS,
                                       generator=torch.Generator("cuda").manual_seed(k)).images[0])
            images = {"paste": blend(bg, fg, alpha), "inpaint": blend(painted, fg, alpha)}
            times["inpaint"] += time.perf_counter() - t
            t = time.perf_counter()
            masks = [m for _, m in gts]
            p_bg = judge.ask([Image.fromarray(bg), Image.fromarray(gray_boxes(images["inpaint"], masks))], bg_instruction, yes_no)
            times["judge"] += time.perf_counter() - t
            t = time.perf_counter()
            for v, img in images.items():
                Image.fromarray(img).save(out_dir / v / f"{k:05d}.jpg", quality=JPEG_QUALITY)
            f.write(json.dumps({"id": k, "context": context, "description": di,
                                "objects": [{"class": nm, "object_id": layout[i].obj.id, "segmentation": mask_to_rle(m)}
                                            for (nm, m), i in zip(gts, kept)],
                                "filter2_yes": {"paste": p_bg[0]["yes"], "inpaint": p_bg[1]["yes"]}}) + "\n")
            f.flush()
            times["save"] += time.perf_counter() - t
            if (k + 1) % 100 == 0 or k == n - 1:
                m = k + 1 - done
                print(f"場面 {k} 1 場面あたり秒: " + " ".join(f"{a} {b / m:.2f}" for a, b in times.items()), flush=True)


def main() -> None:
    """引数で選んだ段(--stage)を順に動かす。"""
    global OUT, N_DESC
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["descriptions", "objects", "filter1", "combos", "place_descriptions", "scenes",
                                            "all"], default="all")
    parser.add_argument("--n-scenes", type=int, default=10000)
    parser.add_argument("--n-desc", type=int, default=N_DESC, help="クラスごとの物体の数(DESC_CHUNK の倍数)")
    parser.add_argument("--name", default=OUT.name, help="datasets/synthetic/ の下の保存先の名前(動作確認では別の名前にする)")
    parser.add_argument("--scenes-dir", default="scenes", help="場面の保存先のフォルダ名(試しに少し作るときは別の名前にする)")
    args = parser.parse_args()
    OUT, N_DESC = OUT.parent / args.name, args.n_desc
    # 段ごとに使うモデルを GPU から下ろす(並べて動かす学習の GPU のメモリを空けるため)
    if args.stage in ("descriptions", "all"):
        stage_descriptions()
        unload()
    if args.stage in ("objects", "all"):
        stage_objects()
        gc.collect(), torch.cuda.empty_cache()
    if args.stage in ("filter1", "all"):
        stage_filter1()
        unload()
    if args.stage in ("combos", "all"):
        stage_combos()
    if args.stage in ("place_descriptions", "all"):
        stage_place_descriptions()
    if args.stage in ("scenes", "all"):
        stage_scenes(args.n_scenes, OUT / args.scenes_dir)


if __name__ == "__main__":
    main()

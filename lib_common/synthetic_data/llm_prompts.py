"""手元の ollama の LLM で、物体の多様な説明文を作る(画像生成のプロンプトの多様さのため)。

ollama の HTTP API(http://localhost:11434)を使う。手元で動くので、ネットワークには出ない。
説明文は物体そのものだけ(背景・地面・ほかの物体は書かせない)。背景・影なしの文面は prompts.object_prompt_from_description で足す。
"""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"


def unload(model: str = DEFAULT_MODEL) -> None:
    """ollama のモデルを GPU から下ろす(keep_alive 0 の空の依頼。ほかの GPU の仕事と並べて動かすとき、使い終わったら呼ぶ)。

    ollama は既定で最後の依頼から 5 分モデルを GPU に置いたままにするため、そのままだと後の仕事の GPU のメモリを圧迫する。

    Args:
        model: 下ろす ollama のモデル名。
    """
    req = {"model": model, "keep_alive": 0}
    urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}), timeout=60).read()

INSTRUCTION = ("Write {n} diverse, short visual descriptions of a single {name} for photorealistic image generation. "
               "Vary the type, color, material, size or age, pose or state, as much as is natural for a {name}. "
               "Each must describe exactly one {name} with its whole body visible. "
               "Do not describe any background, ground, floor, scene, lighting, or other objects. "
               'Return JSON: {{"descriptions": ["...", ...]}}')


# 物の種類ごとの、変える観点(動物のポーズのように、その種類に意味のある観点だけ)
VARIATION_AXES = {
    "animal": "breed or species, age (young/adult/old), coat or feather color and pattern, pose, action, viewpoint",
    "person": "age, gender, body type, clothing style and color, accessories carried, pose, action, viewpoint",
    "vehicle": "model type, body style, color, era (modern/vintage), condition (new/dirty/worn), viewpoint",
    "furniture": "style (modern/antique/rustic/industrial...), material, color, shape, size, viewpoint",
    "container": "material (glass/plastic/metal/ceramic...), shape, size, color, label or no label, contents, cap, viewpoint",
    "plant": "plant species, size, leaf shape and color, flowers or not, pot material, pot color and shape, viewpoint",
    "electronics": "screen size, era (modern/old CRT), frame color and material, stand type, screen on or off, viewpoint",
}

# クラスごとの、物の種類と、クラスの範囲(COCO のクラスに合わせ、紛らわしい物を含めない)
CLASS_SPECS = {
    "airplane": ("vehicle", "a fixed-wing airplane; not a helicopter, drone, rocket or toy"),
    "bicycle": ("vehicle", "a pedal bicycle; not a motorcycle or scooter"),
    "bird": ("animal", "a real bird species"),
    "boat": ("vehicle", "a boat or ship on water use; not a submarine"),
    "bottle": ("container", "a bottle with a narrow neck; not a can, cup, jar or vase"),
    "bus": ("vehicle", "a bus for passengers; not a truck or van"),
    "car": ("vehicle", "a passenger car (sedan, SUV, hatchback, sports car, pickup); not a bus or truck"),
    "cat": ("animal", "a domestic cat; not a big cat like a lion or tiger"),
    "chair": ("furniture", "a chair for one person; not a sofa, bench or stool"),
    "cow": ("animal", "a cow, bull or calf (cattle)"),
    "dining table": ("furniture", "a dining table; not a desk, coffee table or counter"),
    "dog": ("animal", "a domestic dog; not a wolf or fox"),
    "horse": ("animal", "a horse, pony or foal; not a donkey or zebra"),
    "motorcycle": ("vehicle", "a motorcycle or motor scooter; not a bicycle"),
    "person": ("person", "one real human person"),
    "potted plant": ("plant", "a plant growing in a pot or planter"),
    "sheep": ("animal", "a sheep or lamb; not a goat"),
    "couch": ("furniture", "a sofa or couch for two or more people; not a single armchair or bed"),
    "train": ("vehicle", "a train, locomotive, tram or subway train on rails"),
    "tv": ("electronics", "a television set; not a computer monitor, laptop or phone"),
}

INSTRUCTION_V2 = ("Write {n} diverse, short visual descriptions of a single {name} for photorealistic image generation. "
                  "The object must be: {scope}. Only real, ordinary objects that exist in the real world; nothing fantasy or toy-like. "
                  "Vary these aspects across the descriptions: {axes}. "
                  "Each must describe exactly one {name} with its whole body visible. "
                  "Do not describe any background, ground, floor, scene, lighting, or other objects. "
                  'Return JSON: {{"descriptions": ["...", ...]}}')


def describe_v2(name: str, n: int, seed: int = 0, model: str = DEFAULT_MODEL, timeout: float = 300) -> list[str]:
    """describe と同じだが、クラスの範囲(CLASS_SPECS)と、物の種類に合った変える観点(VARIATION_AXES)を LLM に渡す。

    Args:
        name: クラス名。
        n: 作る説明文の数。
        seed: LLM の乱数の種。
        model: ollama のモデル名。
        timeout: 1 回の依頼の待ち時間の上限(秒)。

    Returns:
        説明文のリスト(長さ n)。
    """
    kind, scope = CLASS_SPECS[name]
    return describe(name, n, seed, model, timeout,
                    instruction=INSTRUCTION_V2.replace("{scope}", scope).replace("{axes}", VARIATION_AXES[kind]))


# こちらで割り当てる観点(LLM に任せると偏る観点。llm_v2 では person がほぼ全部「まっすぐ立って正面」になった)
VIEWPOINTS = ["side view", "front view", "front three-quarter view", "rear three-quarter view", "view from slightly above",
              "back view"]
POSES = {
    "animal": ["standing", "walking", "running", "sitting", "lying down", "jumping", "looking back over its shoulder",
               "eating or sniffing downward", "stretching"],
    "person": ["standing", "walking", "running", "sitting cross-legged", "jumping", "waving one hand", "crouching",
               "carrying a bag", "dancing", "bending forward"],
}


def assignments(name: str, n: int, seed: int) -> list[str]:
    """n 個の説明文それぞれに、向き(と動物・人ならポーズ・動作)を割り当てる。一覧を並べ替えて順に配り、偏りをなくす。

    Args:
        name: クラス名。
        n: 説明文の数。
        seed: 並べ替えの乱数の種。

    Returns:
        説明文ごとの割り当ての文(長さ n)。
    """
    import random
    rng = random.Random(seed)
    kind = CLASS_SPECS[name][0]

    def spread(items: list[str]) -> list[str]:
        """一覧を並べ替えて、n 個になるまで繰り返す(どの項目もほぼ同じ回数になる)。

        Args:
            items: 割り当てる項目の一覧。

        Returns:
            長さ n の項目の並び。
        """
        out = []
        while len(out) < n:
            out += rng.sample(items, len(items))
        return out[:n]

    views = spread(VIEWPOINTS)
    if kind in POSES:
        return [f"pose/action: {p}; viewpoint: {v}" for p, v in zip(spread(POSES[kind]), views)]
    return [f"viewpoint: {v}" for v in views]


INSTRUCTION_V3 = ("Write {n} short visual descriptions of a single {name} for photorealistic image generation, one for each "
                  "numbered requirement below, in the same order. "
                  "The object must be: {scope}. Only real, ordinary objects that exist in the real world; nothing fantasy or toy-like. "
                  "Each description must follow its requirement exactly, and the descriptions must differ from each other in: {axes}. "
                  "Each must describe exactly one {name} with its whole body visible. "
                  "Do not describe any background, ground, floor, scene, lighting, or other objects.\n"
                  "{requirements}\n"
                  'Return JSON: {{"descriptions": ["...", ...]}}')


def describe_v3(name: str, n: int, seed: int = 0, model: str = DEFAULT_MODEL, timeout: float = 300) -> list[str]:
    """describe_v2 に加え、向き(動物・人はポーズ・動作も)をこちらで割り当てて、説明文ごとに守らせる。

    Args:
        name: クラス名。
        n: 作る説明文の数。
        seed: 割り当てと LLM の乱数の種。
        model: ollama のモデル名。
        timeout: 1 回の依頼の待ち時間の上限(秒)。

    Returns:
        説明文のリスト(長さ n。i 番目は assignments の i 番目の割り当てに従う)。
    """
    kind, scope = CLASS_SPECS[name]
    reqs = assignments(name, n, seed)
    out: list[str] = []
    for attempt in range(10):  # 足りなかった分の割り当てだけを、続きとして作らせる(i 番目の説明文は i 番目の割り当てに従う)
        need = reqs[len(out):]
        req = {"model": model, "stream": False, "think": False, "format": "json",
               "options": {"temperature": 1.0, "seed": seed + attempt, "num_predict": 8192, "num_ctx": 16384},
               "prompt": INSTRUCTION_V3.format(
                   n=len(need), name=name, scope=scope, axes=VARIATION_AXES[kind],
                   requirements="\n".join(f"{i + 1}. {r}" for i, r in enumerate(need)))}
        with urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, json.dumps(req).encode(),
                                                           {"Content-Type": "application/json"}), timeout=timeout) as r:
            text = json.load(r)["response"]
        try:
            got = [d.strip() for d in json.loads(text)["descriptions"] if isinstance(d, str) and d.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # 形が崩れた答え(長すぎて途中で切れたものなど)は捨てる
        out += got[: len(need)]
        if len(out) >= n:
            return out[:n]
    raise RuntimeError(f"{name} の説明文を {n} 個そろえて作れなかった({len(out)} 個)")


def describe(name: str, n: int, seed: int = 0, model: str = DEFAULT_MODEL, timeout: float = 300,
             instruction: str = INSTRUCTION) -> list[str]:
    """クラス名の物体の説明文を n 個作る(足りなければ作り直して足す)。

    Args:
        name: クラス名(例 "dog")。
        n: 作る数。
        seed: LLM の乱数の種(同じなら同じ文が返る)。
        model: ollama のモデル名。
        timeout: 1 回の問い合わせの待ち時間の上限(秒)。
        instruction: LLM への指示の文面({n} と {name} を含む)。

    Returns:
        説明文のリスト(長さ n)。
    """
    out: list[str] = []
    attempt = 0
    while len(out) < n:
        req = {"model": model, "stream": False, "think": False, "format": "json",
               "options": {"temperature": 1.0, "seed": seed + attempt},
               "prompt": instruction.format(n=n - len(out), name=name)}
        with urllib.request.urlopen(urllib.request.Request(OLLAMA_URL, json.dumps(req).encode(),
                                                           {"Content-Type": "application/json"}), timeout=timeout) as r:
            text = json.load(r)["response"]
        try:
            out += [d.strip() for d in json.loads(text)["descriptions"] if isinstance(d, str) and d.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # 形が崩れた答えは捨てて、作り直す
        attempt += 1
        if attempt > 5:
            raise RuntimeError(f"{name} の説明文を {n} 個作れなかった({len(out)} 個)")
    return out[:n]

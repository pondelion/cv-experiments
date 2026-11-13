"""合成データの画像生成に使うプロンプト(固定の文面の組み合わせ)。

LLM で多様にする前の、基準になる素直な文面。クラスの名前は COCO の名前(lib_common/classes.py)。
"""
import random

VIEWS = ["side view", "front view", "three-quarter view"]

# 物体 1 つだけの画像(切り貼り用)。背景を無地にして、色の差で抜き出せるようにする
OBJECT_TEMPLATE = ("a photo of a single {name}, {view}, the whole {name} is fully visible in the center, "
                   "isolated on {background}, no shadow")
BACKGROUNDS = {
    "green": "a solid pure green chroma key background",
    "white": "a plain pure white background",
    "magenta": "a solid pure magenta background",
}
# クラスごとの背景の色。基本はグリーンバック(白・灰色の物体も抜ける)。緑になりやすいクラスだけ、緑と離れたマゼンタにする
BACKGROUND_FOR = {"potted plant": "magenta"}


def background_for(name: str) -> str:
    """クラスに使う背景の色を返す。

    Args:
        name: クラス名。

    Returns:
        BACKGROUNDS のキー(既定は "green")。
    """
    return BACKGROUND_FOR.get(name, "green")
# 影を描かせないための文面(宙に浮かせ、地面をなくし、照明を平らにする)
NO_SHADOW_TEMPLATE = ("a photo of a single {name}, {view}, the whole {name} is fully visible in the center, "
                      "floating in mid-air, no ground, no floor, flat even lighting, no shadow, isolated on {background}")

# クラスごとの、そのクラスがいそうな場所(背景の生成・シーンの生成用)
CONTEXTS = {
    "person": ["city street", "park", "living room"], "bicycle": ["city street", "park"],
    "car": ["city street", "parking lot"], "motorcycle": ["city street", "parking lot"],
    "airplane": ["airport runway", "sky"], "bus": ["city street"], "train": ["railway station"],
    "boat": ["harbor", "lake"], "bird": ["park", "garden"], "cat": ["living room", "garden"],
    "dog": ["living room", "park"], "horse": ["farm field"], "sheep": ["farm field"], "cow": ["farm field"],
    "bottle": ["kitchen", "dining room"], "chair": ["living room", "dining room"],
    "dining table": ["dining room", "kitchen"], "potted plant": ["living room", "garden"],
    "couch": ["living room"], "tv": ["living room"],
}

# 物体を置く前の、何もない背景
BACKGROUND_TEMPLATE = "a photo of an empty {context}, nobody, no animals, no vehicles"

# 物体を置いた画像の、物体以外の部分を描くとき(物体は描かれているので、場所だけを伝える)
INPAINT_BACKGROUND_TEMPLATE = "a photo taken in a {context}"

# 背景に、学習するクラスの物体を 1 つも描かないよう、全クラスを名前で否定する文面(蒸留版のモデルはネガティブプロンプトが効かないため)
NO_CLASS_TEMPLATE = "a photo of an empty {context}. There is no {classes} in the image."


def background_prompt(description: str, class_names: list[str]) -> str:
    """場所の説明文(scene_combos.place_descriptions)の背景のプロンプトに、全クラスを名前で否定する文を足す。

    例: "a photo of A sunlit stone-walled kitchen .... There is no person, no bicycle, ... and no tv in the image."

    Args:
        description: 場所の説明文。
        class_names: 否定するクラス名の並び。

    Returns:
        プロンプトの文。
    """
    negation = no_class_prompt("X", class_names).split(". ", 1)[1]  # "There is no ... in the image."
    return f"a photo of {description.rstrip('.')}. {negation}"


def no_class_prompt(context: str, class_names: list[str]) -> str:
    """全クラスを名前で否定した背景のプロンプト(例: "... There is no person, no bicycle, ... and no tv in the image.")。

    Args:
        context: 場所。
        class_names: 否定するクラス名の並び。

    Returns:
        プロンプトの文。
    """
    items = [f"no {n}" for n in class_names]
    return NO_CLASS_TEMPLATE.format(context=context, classes=", ".join(items[:-1]).removeprefix("no ") + " and " + items[-1])

# 文章からシーンを自由に生成するとき
SCENE_TEMPLATE = "a photo of {objects} in a {context}"


# クラスごとの属性の一覧(種類・色・ポーズや状態)。組み合わせて、多様な物体の文面を作る
ATTRIBUTES = {
    "airplane": {"type": ["passenger jet", "small propeller plane", "cargo plane", "fighter jet", "seaplane"],
                 "color": ["white", "silver", "red and white", "blue", "yellow"], "pose": ["in flight", "banking to the left", "climbing"]},
    "bicycle": {"type": ["road bike", "mountain bike", "city bike with a basket", "kids bike", "BMX bike"],
                "color": ["black", "red", "blue", "white", "green"], "pose": ["standing upright", "leaning slightly"]},
    "bird": {"type": ["sparrow", "parrot", "pigeon", "seagull", "crow", "owl", "duck"],
             "color": ["brown", "colorful", "grey", "black", "white"], "pose": ["perched", "with wings spread", "flying"]},
    "boat": {"type": ["sailboat", "motorboat", "rowing boat", "fishing boat", "kayak"],
             "color": ["white", "red", "blue", "wooden brown", "yellow"], "pose": ["seen at an angle"]},
    "bottle": {"type": ["wine bottle", "plastic water bottle", "beer bottle", "glass milk bottle", "soda bottle"],
               "color": ["clear", "green glass", "brown glass", "blue", "transparent with a label"], "pose": ["standing upright", "lying on its side"]},
    "bus": {"type": ["city bus", "double-decker bus", "school bus", "coach bus", "minibus"],
            "color": ["red", "yellow", "white", "blue", "green and white"], "pose": ["seen at an angle"]},
    "car": {"type": ["sedan", "SUV", "hatchback", "sports car", "pickup truck", "vintage car"],
            "color": ["red", "black", "white", "silver", "blue"], "pose": ["seen at an angle"]},
    "cat": {"type": ["tabby cat", "black cat", "white cat", "siamese cat", "orange cat", "kitten"],
            "color": ["striped", "spotted", "solid colored"], "pose": ["sitting", "lying down", "stretching", "walking", "curled up"]},
    "chair": {"type": ["wooden dining chair", "office chair", "plastic garden chair", "armchair", "folding chair"],
              "color": ["brown", "black", "white", "red", "grey"], "pose": ["seen at an angle"]},
    "cow": {"type": ["holstein cow", "brown cow", "highland cow", "calf", "black angus cow"],
            "color": ["black and white", "brown", "reddish", "black"], "pose": ["standing", "walking", "lying down", "grazing"]},
    "dining table": {"type": ["round wooden dining table", "long rectangular dining table", "glass-top dining table", "small square table"],
                     "color": ["dark wood", "light wood", "white", "black"], "pose": ["seen from slightly above"]},
    "dog": {"type": ["golden retriever", "labrador", "dachshund", "husky", "poodle", "bulldog", "chihuahua", "german shepherd"],
            "color": ["golden", "black", "white", "brown", "spotted"], "pose": ["sitting", "standing", "running", "lying down", "jumping"]},
    "horse": {"type": ["thoroughbred horse", "pony", "draft horse", "foal", "arabian horse"],
              "color": ["brown", "black", "white", "grey", "chestnut", "spotted"], "pose": ["standing", "walking", "galloping", "rearing"]},
    "motorcycle": {"type": ["sport bike", "cruiser motorcycle", "scooter", "dirt bike", "touring motorcycle"],
                   "color": ["red", "black", "blue", "white", "orange"], "pose": ["standing upright", "on its kickstand"]},
    "person": {"type": ["young woman", "old man", "child", "teenage boy", "middle-aged woman", "man in a suit"],
               "color": ["wearing casual clothes", "wearing a coat", "wearing sportswear", "wearing a dress"],
               "pose": ["standing", "walking", "running", "sitting", "waving", "jumping"]},
    "potted plant": {"type": ["small succulent in a pot", "large leafy plant in a pot", "flowering plant in a pot", "cactus in a pot", "small tree in a pot"],
                     "color": ["green with a terracotta pot", "green with a white pot", "with red flowers", "with yellow flowers"], "pose": ["standing"]},
    "sheep": {"type": ["woolly sheep", "freshly sheared sheep", "lamb", "black-faced sheep"],
              "color": ["white", "cream", "grey", "black"], "pose": ["standing", "walking", "lying down", "grazing"]},
    "couch": {"type": ["three-seat sofa", "two-seat loveseat", "L-shaped sectional sofa", "leather sofa", "chesterfield sofa"],
              "color": ["grey", "brown", "blue", "beige", "green", "black"], "pose": ["seen at an angle"]},
    "train": {"type": ["high-speed train", "steam locomotive", "subway train", "freight train", "tram"],
              "color": ["white", "red", "blue", "black", "silver"], "pose": ["seen at an angle"]},
    "tv": {"type": ["flat screen television", "old CRT television", "large wall-mounted TV", "small portable TV"],
           "color": ["black", "silver", "white"], "pose": ["on a stand", "with the screen turned on", "with the screen turned off"]},
}


def attribute_description(name: str, rng: random.Random) -> str:
    """属性の一覧から 1 つずつ選んで、物体の説明を作る(例: "a brown dachshund, spotted, running")。

    Args:
        name: クラス名(ATTRIBUTES のキー)。
        rng: 乱数。

    Returns:
        物体の説明の文。
    """
    a = ATTRIBUTES[name]
    return f"a {rng.choice(a['color'])} {rng.choice(a['type'])}, {rng.choice(a['pose'])}"


def object_prompt_from_description(description: str, name: str, rng: random.Random, add_view: bool = True) -> str:
    """物体の説明(属性の組み合わせ・LLM の文)を、単一の物体・影なし・無地の背景の文面に入れる。

    Args:
        description: 物体の説明の文。
        name: クラス名(背景の色を決める)。
        rng: 乱数(向きの言葉を選ぶ)。
        add_view: False なら向きの言葉を足さない(説明文の中に向きが書いてあるとき)。

    Returns:
        画像生成のプロンプトの文。
    """
    view = f"{rng.choice(VIEWS)}, " if add_view else ""
    return (f"a photo of a single {name}: {description}, {view}the whole {name} is fully visible in the center, "
            f"floating in mid-air, no ground, no floor, flat even lighting, no shadow, "
            f"isolated on {BACKGROUNDS[background_for(name)]}")


def object_prompt(name: str, rng: random.Random, background: str = "green", no_shadow: bool = False) -> str:
    """物体 1 つだけの画像のプロンプト。

    Args:
        name: クラス名。
        rng: 乱数(向きの言葉を選ぶ)。
        background: BACKGROUNDS のキー。
        no_shadow: True なら影を描かせない文面にする。

    Returns:
        画像生成のプロンプトの文。
    """
    template = NO_SHADOW_TEMPLATE if no_shadow else OBJECT_TEMPLATE
    return template.format(name=name, view=rng.choice(VIEWS), background=BACKGROUNDS[background])


def context_for(names: list[str], rng: random.Random) -> str:
    """複数のクラスに共通しやすい場所を 1 つ選ぶ(共通のものがなければ、最初のクラスの場所)。

    Args:
        names: クラス名の並び。
        rng: 乱数。

    Returns:
        場所(CONTEXTS の値の 1 つ)。
    """
    common = set(CONTEXTS[names[0]]).intersection(*(CONTEXTS[n] for n in names[1:]))
    return rng.choice(sorted(common) or CONTEXTS[names[0]])


def scene_prompt(names: list[str], rng: random.Random) -> str:
    """複数のクラスの物体が写るシーンのプロンプト(例: "a photo of a dog and a person in a park")。

    Args:
        names: クラス名の並び。
        rng: 乱数(場所を選ぶ)。

    Returns:
        画像生成のプロンプトの文。
    """
    objs = [f"a {n}" for n in names]
    objects = objs[0] if len(objs) == 1 else ", ".join(objs[:-1]) + " and " + objs[-1]
    return SCENE_TEMPLATE.format(objects=objects, context=context_for(names, rng))

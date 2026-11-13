"""CV のタスクで共通の処理(データセットの読み込み・クラスの定義・速度の測り方など)。

各 exp のスクリプトは、exp の中の `common` パッケージを import するとリポジトリ直下が import の探索パスに入り、
`from lib_common.xxx import ...`(CV で共通)、`from segmentation.lib_common.xxx import ...`(セグメンテーションで共通)の
ように、置き場所どおりの名前で読み込める。
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # lib_common → リポジトリ直下
DATASETS_DIR = REPO_ROOT / "datasets"

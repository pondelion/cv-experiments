"""exp00201 固有の処理。

ここでリポジトリ直下を import の探索パスに追加するので、各スクリプトは `import common` の後で
`from lib_common.xxx import ...` や `from lib_common.synthetic_data.xxx import ...` と書ける。
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]  # common → exp00201 → instance_segmentation → segmentation → リポジトリ直下
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
RUNS_DIR = EXP_DIR / "runs"  # 生成した画像・マスク(git の対象外)
SIZE = 768  # 生成する画像の一辺の画素数

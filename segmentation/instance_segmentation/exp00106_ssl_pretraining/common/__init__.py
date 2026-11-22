"""exp00106 固有の処理。

ここでリポジトリ直下を import の探索パスに追加するので、各スクリプトは `import common` の後で `from lib_common.xxx import ...` と書ける。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # common → exp00106 → instance_segmentation → segmentation → リポジトリ直下
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # 重み・ログ(git の対象外)
SYNTH_IMAGES = [REPO_ROOT / "datasets/synthetic/voc20_v1" / d for d in ("objects", "scenes/paste", "scenes/inpaint")]
KINETICS_TRAIN = REPO_ROOT / "datasets/kinetics400/train"  # VIPSeg と重なる動画は除いてある
KINETICS_BROKEN = REPO_ROOT / "datasets/kinetics400/broken.txt"  # 開けない動画(datasets/kinetics400/find_broken.py)


def kinetics_files() -> list[str]:
    """Kinetics-400 の学習に使う動画のパス(開けない動画を除く)。"""
    broken = set(KINETICS_BROKEN.read_text().split()) if KINETICS_BROKEN.exists() else set()
    return sorted(str(f) for f in KINETICS_TRAIN.glob("*.mp4") if f.name not in broken)

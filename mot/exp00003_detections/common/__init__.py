"""exp00003 固有の処理。

ここでリポジトリ直下を import の探索パスに追加するので、各スクリプトは `import common` の後で
`from lib_common.xxx import ...`(CV で共通)や `from mot.lib_common.xxx import ...`(mot/ で共通)と書ける。
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]  # common → exp00001 → mot → リポジトリ直下
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
RUNS_DIR = EXP_DIR / "runs"  # 保存した検出(git の対象外)

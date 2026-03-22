from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcrypto.cli import run_experiments_main


if __name__ == "__main__":
    run_experiments_main()

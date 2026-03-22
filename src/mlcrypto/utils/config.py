from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

import sys
import yaml
from pathlib import Path
from typing import Optional

def load_config(path: Optional[Path] = None) -> dict:
    """
    Loads and parses config.yaml. Shared by every module in the pipeline so
    there's exactly one place that knows how to read the config file.
 
    Args:
        path: Explicit path to config.yaml. If None, resolves to
              <project_root>/config/config.yaml, assuming this file lives
              at <project_root>/<something>/config_loader.py (one level
              deeper than BASE_DIR). Adjust the .parent chain if you place
              this file elsewhere in your project structure.
    """

    if path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        path = base_dir / "config" / "config.yaml"

        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"FATAL ERROR: Failed to parse {path}. Check your syntax. Details: {e}")
            sys.exit(1)
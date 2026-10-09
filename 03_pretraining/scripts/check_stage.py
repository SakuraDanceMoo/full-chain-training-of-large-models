"""Check this stage against its declared artifact gate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_repository import check_stage
if __name__ == "__main__":
    sys.exit(check_stage(Path(__file__).resolve().parents[1]))

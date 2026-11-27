"""Make the flat src modules available to tests and command-line entry points."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
(Path(__file__).resolve().parents[1] / 'outputs' / 'cache').mkdir(parents=True, exist_ok=True)
sys.pycache_prefix=str(Path(__file__).resolve().parents[1] / 'outputs' / 'cache' / 'python')

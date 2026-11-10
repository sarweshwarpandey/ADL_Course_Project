import os
from pathlib import Path
import runpy

# Wrapper that calls the root plot_results.py so running from either location saves to ./outputs/plots/
root_script = Path(__file__).resolve().parent.parent.parent / "plot_results.py"
os.chdir(root_script.parent)
runpy.run_path(str(root_script), run_name="__main__")

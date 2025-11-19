import sys
from pathlib import Path

# Import existing modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.turbidity_viz import create_state_comparison_plot

root = Path("./turbidity_results")
json_files = list(root.glob("**/*.turbidity.json"))
print(f"found {len(json_files)} json files")
print("Current working directory:", Path.cwd())
print("Looking for JSON in:", Path("./").resolve())

out_plot = Path("./comparison_plots")
create_state_comparison_plot(
    json_paths=json_files,
    metric="std",
    output_path=out_plot,
    state_order=["stable", "gel", "phase_separated", "only_air"],
)
print("saved:", out_plot)

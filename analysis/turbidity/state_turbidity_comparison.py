import sys
from pathlib import Path

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.turbidity_viz import create_state_comparison_plot

root = Path("turbidity_results")
json_files = list(root.glob("**/*.turbidity.json"))

out_plot = Path("turbidity_results/mean_by_state.png")
create_state_comparison_plot(
    json_paths=json_files,
    metric="mean",
    output_path=out_plot,
    state_order=["stable", "gel", "phase_separated", "only_air"],
)
print("saved:", out_plot)

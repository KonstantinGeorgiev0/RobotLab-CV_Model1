"""
Visualization functions for turbidity analysis and detections.
"""

import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

matplotlib.use('Agg')  # Non-interactive backend

from config import CLASS_IDS, VISUALIZATION_PARAMS, TURBIDITY_PARAMS
from robotlab_utils.bbox_utils import yolo_line_to_xyxy
from ultralytics.utils.plotting import Annotator, colors
from matplotlib.patches import Patch


def save_turbidity_plot(
        path,
        v_norm,
        turbidity_profile,
        excluded_info: Optional[Dict[str, Any]] = None,
        out_dir: Optional[Path] = None,
        change_events: Optional[List[Dict[str, Any]]] = None,
        suffix: str = ".turbidity.png",
        use_normalized_height: bool = True,
):
    """
    Save a combined turbidity and gradient analysis plot.

    Args:
        path: Path to the source image
        v_norm: Normalized turbidity profile (1D numpy array)
        excluded_info: Dict with excluded region info
        out_dir: Directory to save output plot
        change_events: List of sudden brightness change events
        suffix: Suffix for output plot file name
        use_normalized_height: Whether to use normalized height for y-axis

    Returns:
        str: Path to saved plot
    """
    # analysis_region
    analysis_height = TURBIDITY_PARAMS['analysis_height']

    if use_normalized_height:
        z = np.linspace(0, 1, len(v_norm))
        suffix = suffix.replace(".png", "_normalized.png")
    else:
        z = np.linspace(0, analysis_height, len(v_norm))

    # ensure dir exists
    file_dir = Path(out_dir) if out_dir else None
    file_dir.mkdir(parents=True, exist_ok=True)

    if file_dir is not None:
        filename = Path(path).stem + suffix
        out_path = Path(file_dir) / filename
    else:
        out_path = Path(path).with_suffix(suffix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=120)

    # turbidity profile
    ax1.plot(v_norm, z, 'b-', linewidth=2)

    if excluded_info:
        top_frac = excluded_info['excluded_top_fraction']
        bottom_frac = excluded_info['excluded_bottom_fraction']

        if use_normalized_height:
            ax1.axhspan(0, top_frac, alpha=0.2, color='red')
            ax1.axhspan(1 - bottom_frac, 1, alpha=0.2, color='orange')
        else:
            top_px = analysis_height * top_frac
            bot_px = analysis_height * bottom_frac
            ax1.axhspan(0, top_px, alpha=0.2, color='red')
            ax1.axhspan(analysis_height - bot_px, analysis_height, alpha=0.2, color='orange')

    if change_events:
        inc_color = 'lime'
        dec_color = 'magenta'
        has_inc = False
        has_dec = False

        for ev in change_events:
            start_px = ev.get('start_absolute') or (ev.get('start_norm', 0) * analysis_height)
            end_px = ev.get('end_absolute') or (ev.get('end_norm', 1) * analysis_height)
            start_norm = ev.get('start_norm', 0)
            end_norm = ev.get('end_norm', 1)

            if 'direction' not in ev:
                # guess direction from brightness change
                delta = ev.get('intensity_change', 0)
                direction = 'increasing' if delta > 0 else 'decreasing'
            else:
                direction = ev['direction']

            color = inc_color if direction == 'increasing' else dec_color
            if direction == 'increasing':
                has_inc = True
            else:
                has_dec = True

            # highlight transition zone
            if use_normalized_height:
                ax1.axhspan(start_norm, end_norm, alpha=0.3, color=color)
            else:
                ax1.axhspan(start_px, end_px, alpha=0.3, color=color)

        # legend
        legend_patches = []
        if has_inc:
            legend_patches.append(Patch(facecolor=inc_color, alpha=0.3, label='Sudden increase'))
        if has_dec:
            legend_patches.append(Patch(facecolor=dec_color, alpha=0.3, label='Sudden decrease'))
        if legend_patches:
            ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)

    # ax1.invert_yaxis()
    ax1.set_xlabel("Brightness (normalized)")
    if use_normalized_height:
        ax1.set_ylabel("Normalized Height")
    else:
        ax1.set_ylabel("Height (pixels)")
    ax1.set_title("Brightness Profile", fontsize=10)

    # gradient analysis
    gradient = np.abs(np.gradient(v_norm))
    if use_normalized_height:
        z_grad = np.linspace(0, 1, len(gradient))
    else:
        z_grad = np.linspace(0, analysis_height, len(gradient))

    ax2.plot(gradient, z_grad, 'g-', linewidth=2)

    # threshold and peak detection
    threshold = max(np.mean(gradient) + 2.5 * np.std(gradient), 0.06)
    ax2.axvline(threshold, color='r', linestyle='--', alpha=0.5)

    peak_positions = np.where(gradient > threshold)[0]
    if len(peak_positions) > 0:
        # mark centers of sudden changes
        if change_events:
            for ev in change_events:
                start = ev.get('start_norm', ev.get('start_normalized'))
                end = ev.get('end_norm', ev.get('end_normalized'))
                if start is None or end is None:
                    continue

                center_y = (start + end) / 2.0
                idx = int(center_y * (len(gradient) - 1))
                idx = max(0, min(len(gradient) - 1, idx))
                direction = ev.get('direction', 'increasing')
                marker = '^' if direction == 'increasing' else 'v'

                ax2.scatter(
                    gradient[idx],
                    z_grad[idx],
                    s=30,
                    marker=marker,
                    edgecolor='k',
                    facecolor='none',
                    zorder=10,
                )

        min_sep = int(len(v_norm) * 0.1)
        groups, current = [], [peak_positions[0]]
        for i in range(1, len(peak_positions)):
            if peak_positions[i] - peak_positions[i - 1] < min_sep:
                current.append(peak_positions[i])
            else:
                groups.append(current)
                current = [peak_positions[i]]
        groups.append(current)

        merged_peaks = [g[np.argmax(gradient[g])] for g in groups]
        ax2.scatter(
            gradient[merged_peaks],
            z_grad[merged_peaks],
            c='red', s=25, zorder=10,
            label='Significant peaks')

    if use_normalized_height:
        ax1.set_ylim(1, 0)
    else:
        ax1.set_ylim(analysis_height, 0)
    ax2.invert_yaxis()

    ax2.set_ylim(ax2.get_ylim())
    ax2.set_xlabel("Gradient Magnitude")
    ax2.set_ylabel("Normalized Height" if use_normalized_height else "Height (pixels)")
    ax2.set_title("Gradient Analysis", fontsize=10)

    plt.suptitle(Path(path).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    return str(out_path)


def save_turbidity_plot_analysis_only(
        path,
        v_norm,
        excluded_info: Dict[str, Any],
        out_dir: Optional[Path] = None,
        suffix: str = ".turbidity_analysis_region.png",
):
    """
    Save a turbidity and gradient plot that only shows the analysis band,
    i.e., from top_exclude_idx to bottom_exclude_idx. Y-axis is normalized
    0–1 within that band.
    """
    top_idx = excluded_info['top_exclude_idx']
    bottom_idx = excluded_info['bottom_exclude_idx']

    if bottom_idx <= top_idx:
        return None

    profile_slice = v_norm[top_idx:bottom_idx]

    # y space
    z = np.linspace(0, 1, len(profile_slice))

    file_dir = Path(out_dir) if out_dir else Path(path).parent
    file_dir.mkdir(parents=True, exist_ok=True)
    out_path = file_dir / (Path(path).stem + suffix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=120)

    # brightness
    ax1.plot(profile_slice, z, 'b-', linewidth=2)
    ax1.invert_yaxis()
    ax1.set_xlabel("Brightness (normalized)")
    ax1.set_ylabel("Normalized Height (analysis only)")
    ax1.set_title("Brightness Profile (analysis region)", fontsize=10)

    # gradient in analysis region
    gradient = np.abs(np.gradient(profile_slice))
    z_grad = np.linspace(0, 1, len(gradient))
    ax2.plot(gradient, z_grad, 'g-', linewidth=2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Gradient Magnitude")
    ax2.set_ylabel("Normalized Height (analysis only)")
    ax2.set_title("Gradient Analysis (analysis region)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return str(out_path)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a turbidity JSON file safely."""
    with open(path, "r") as f:
        return json.load(f)


def create_state_comparison_plot(
        json_paths: List[Path],
        metric: str,
        output_path: Path,
        state_order=None,
        figsize=(7, 4),
):
    """
    Create a comparison plot (e.g., mean turbidity) for each state
    using all turbidity .json files.

    Args:
        json_paths: list of paths to JSON files
        metric: which metric to compare (e.g. 'mean', 'variance', 'std')
        output_path: path to .png output plot
        state_order: list defining order of states on X-axis
        figsize: figure size
    """

    # collect data per state
    data = {}  # state - list of metric values

    for jp in json_paths:
        info = load_json(jp)

        state = info.get("state", "unknown")
        gstats = info.get("global_stats", {})

        if metric not in gstats:
            print(f"[warning] {metric} not in {jp.name}, skipping.")
            continue

        value = gstats.get(metric)

        if state not in data:
            data[state] = []
        data[state].append(value)

    # order states
    if state_order:
        ordered_states = [s for s in state_order if s in data]
    else:
        ordered_states = sorted(data.keys())

    # prepare plotting data
    x = ordered_states
    y = [sum(data[s]) / len(data[s]) for s in ordered_states]
    counts = [len(data[s]) for s in ordered_states]

    # plot
    plt.figure(figsize=figsize)
    bars = plt.bar(x, y, color="skyblue", edgecolor="k")

    # annotate bar values and counts
    for idx, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{y[idx]:.3f}\n(n={counts[idx]})",
            ha="center", va="bottom", fontsize=8
        )

    plt.ylabel(metric)
    plt.xlabel("Vial State")
    plt.title(f"{metric} comparison by state")
    plt.xticks(rotation=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()



def create_detection_visualization(crop_path: Path,
                                  label_path: Path,
                                  output_path: Path,
                                  conf_threshold: float = 0.2) -> Path:
    """
    Create visualization with detection bounding boxes.

    Args:
        crop_path: Path to crop image
        label_path: Path to label file
        output_path: Path to save visualization
        conf_threshold: Minimum confidence to show

    Returns:
        Path to saved visualization
    """
    img = cv2.imread(str(crop_path))
    if img is None:
        return None

    H, W = img.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Class names mapping
    class_names = {
        CLASS_IDS['GEL']: "gel",
        CLASS_IDS['STABLE']: "stable",
        CLASS_IDS['AIR']: "air",
        CLASS_IDS['CAP']: "cap"
    }

    # Initialize annotator
    annotator = Annotator(
        img,
        line_width=VISUALIZATION_PARAMS['line_thickness']
    )

    # Parse and draw detections
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parsed = yolo_line_to_xyxy(line.strip(), W, H)
                if not parsed:
                    continue

                cls_id, box, conf = parsed

                if conf < conf_threshold:
                    continue

                # Prepare label
                label = f"{class_names.get(cls_id, 'unknown')} {conf:.2f}"

                # Draw box
                x1, y1, x2, y2 = map(int, box)
                annotator.box_label(
                    [x1, y1, x2, y2],
                    label,
                    color=colors(cls_id, True)
                )

    # Save result
    result = annotator.result()
    cv2.imwrite(str(output_path), result)

    return output_path


def create_summary_plot(manifest_path: Path,
                       output_dir: Path) -> Path:
    """
    Create summary visualization from manifest data.

    Args:
        manifest_path: Path to manifest file
        output_dir: Directory to save plot

    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.png"

    # Read manifest data
    state_counts = {}
    turbidity_means = []
    turbidity_max_grads = []

    with open(manifest_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())

            # Count states
            state = record.get('vial_state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1

            # Collect turbidity stats
            if 'turbidity_stats' in record:
                turbidity_means.append(record['turbidity_stats'].get('mean', 0))
                turbidity_max_grads.append(record['turbidity_stats'].get('max_gradient', 0))

    # Create subplots
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: State distribution pie chart
    ax1 = plt.subplot(2, 2, 1)
    if state_counts:
        colors_map = {
            'stable': '#2ecc71',
            'gelled': '#e74c3c',
            'phase_separated': '#f39c12',
            'only_air': '#95a5a6',
            'unknown': '#34495e'
        }
        colors_list = [colors_map.get(k, '#666666') for k in state_counts.keys()]

        wedges, texts, autotexts = ax1.pie(
            state_counts.values(),
            labels=state_counts.keys(),
            colors=colors_list,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Vial State Distribution')

    # Plot 2: State counts bar chart
    ax2 = plt.subplot(2, 2, 2)
    if state_counts:
        states = list(state_counts.keys())
        counts = list(state_counts.values())
        bars = ax2.bar(states, counts)

        # Color bars
        for bar, state in zip(bars, states):
            bar.set_color(colors_map.get(state, '#666666'))

        ax2.set_xlabel('Vial State')
        ax2.set_ylabel('Count')
        ax2.set_title('Vial Count by State')
        ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Turbidity distribution
    ax3 = plt.subplot(2, 2, 3)
    if turbidity_means:
        ax3.hist(turbidity_means, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Mean Turbidity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Turbidity Distribution')
        ax3.axvline(np.mean(turbidity_means), color='red',
                   linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(turbidity_means):.3f}')
        ax3.legend()

    # Plot 4: Turbidity scatter plot
    ax4 = plt.subplot(2, 2, 4)
    if turbidity_means and turbidity_max_grads:
        ax4.scatter(turbidity_means, turbidity_max_grads, alpha=0.5)
        ax4.set_xlabel('Mean Turbidity')
        ax4.set_ylabel('Max Gradient')
        ax4.set_title('Turbidity Characteristics')
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Vial Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_batch_comparison_plot(manifest_paths: List[Path],
                                batch_names: List[str],
                                output_path: Path) -> Path:
    """
    Create comparison plot across multiple batches.

    Args:
        manifest_paths: List of manifest file paths
        batch_names: Names for each batch
        output_path: Path to save comparison plot

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_state_data = {}

    for manifest_path, batch_name in zip(manifest_paths, batch_names):
        state_counts = {}

        with open(manifest_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                state = record.get('vial_state', 'unknown')
                state_counts[state] = state_counts.get(state, 0) + 1

        all_state_data[batch_name] = state_counts

    # Plot 1: Stacked bar chart
    ax1 = axes[0, 0]
    states = ['stable', 'gelled', 'phase_separated', 'only_air', 'unknown']
    bottom = np.zeros(len(batch_names))

    colors_map = {
        'stable': '#2ecc71',
        'gelled': '#e74c3c',
        'phase_separated': '#f39c12',
        'only_air': '#95a5a6',
        'unknown': '#34495e'
    }

    for state in states:
        values = [all_state_data[batch].get(state, 0) for batch in batch_names]
        ax1.bar(batch_names, values, bottom=bottom,
               label=state, color=colors_map.get(state, '#666666'))
        bottom += values

    ax1.set_ylabel('Count')
    ax1.set_title('State Distribution by Batch')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Line plot for trends
    ax2 = axes[0, 1]
    for state in ['stable', 'gelled', 'phase_separated']:
        values = [all_state_data[batch].get(state, 0) for batch in batch_names]
        ax2.plot(batch_names, values, marker='o', label=state, linewidth=2)

    ax2.set_ylabel('Count')
    ax2.set_title('State Trends Across Batches')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Percentage distribution
    ax3 = axes[1, 0]
    x = np.arange(len(batch_names))
    width = 0.15

    for i, state in enumerate(['stable', 'gelled', 'phase_separated']):
        totals = [sum(all_state_data[batch].values()) for batch in batch_names]
        percentages = [
            (all_state_data[batch].get(state, 0) / total * 100) if total > 0 else 0
            for batch, total in zip(batch_names, totals)
        ]
        ax3.bar(x + i * width, percentages, width,
               label=state, color=colors_map.get(state, '#666666'))

    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Percentage Distribution by State')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(batch_names, rotation=45)
    ax3.legend()

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    table_data = [['Batch', 'Total', 'Stable', 'Gelled', 'Phase Sep.']]
    for batch in batch_names:
        row = [
            batch,
            sum(all_state_data[batch].values()),
            all_state_data[batch].get('stable', 0),
            all_state_data[batch].get('gelled', 0),
            all_state_data[batch].get('phase_separated', 0)
        ]
        table_data.append([str(x) for x in row])

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle('Batch Comparison Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

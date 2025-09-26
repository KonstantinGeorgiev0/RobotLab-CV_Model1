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

from config import CLASS_IDS, VISUALIZATION_PARAMS
from robotlab_utils.bbox_utils import yolo_line_to_xyxy
from ultralytics.utils.plotting import Annotator, colors

def save_enhanced_turbidity_plot(path, v_norm, excluded_info=None, out_dir=None):
    """
    Save a combined turbidity + gradient analysis plot (two subplots in one image).

    Args:
        path: Path to the source image (used for naming)
        v_norm: Normalized turbidity profile (1D numpy array)
        excluded_info: Dict with excluded region info (can be None)
        out_dir: Directory to save output plot

    Returns:
        str: Path to saved plot
    """
    z = np.linspace(0, 1, len(v_norm))

    if out_dir is not None:
        filename = Path(path).stem + ".turbidity_enhanced.png"
        out_path = Path(out_dir) / filename
    else:
        out_path = Path(path).with_suffix(".turbidity_enhanced.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=120)

    # ---- Plot 1: Turbidity profile ----
    ax1.plot(v_norm, z, 'b-', linewidth=2)
    if excluded_info:
        ax1.axhspan(0, excluded_info['excluded_top_fraction'], alpha=0.2, color='red')
        ax1.axhspan(1 - excluded_info['excluded_bottom_fraction'], 1, alpha=0.2, color='orange')
    ax1.invert_yaxis()
    ax1.set_xlabel("Turbidity (normalized)")
    ax1.set_ylabel("Normalized Height")
    ax1.set_title("Turbidity Profile", fontsize=10)

    # ---- Plot 2: Gradient analysis ----
    gradient = np.abs(np.gradient(v_norm))
    z_grad = np.linspace(0, 1, len(gradient))
    ax2.plot(gradient, z_grad, 'g-', linewidth=2)

    # Threshold & peak detection
    threshold = max(np.mean(gradient) + 2.5 * np.std(gradient), 0.06)
    ax2.axvline(threshold, color='r', linestyle='--', alpha=0.5)

    peak_positions = np.where(gradient > threshold)[0]
    if len(peak_positions) > 0:
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
        ax2.scatter(gradient[merged_peaks], z_grad[merged_peaks],
                    c='red', s=25, label='Significant peaks')

    ax2.invert_yaxis()
    ax2.set_xlabel("Gradient Magnitude")
    ax2.set_ylabel("Normalized Height")
    ax2.set_title("Gradient Analysis", fontsize=10)

    plt.suptitle(Path(path).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    return str(out_path)

#
# def create_detection_visualization(crop_path: Path,
#                                   label_path: Path,
#                                   output_path: Path,
#                                   conf_threshold: float = 0.2) -> Path:
#     """
#     Create visualization with detection bounding boxes.
#
#     Args:
#         crop_path: Path to crop image
#         label_path: Path to label file
#         output_path: Path to save visualization
#         conf_threshold: Minimum confidence to show
#
#     Returns:
#         Path to saved visualization
#     """
#     img = cv2.imread(str(crop_path))
#     if img is None:
#         return None
#
#     H, W = img.shape[:2]
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # Class names mapping
#     class_names = {
#         CLASS_IDS['GEL']: "gel",
#         CLASS_IDS['STABLE']: "stable",
#         CLASS_IDS['AIR']: "air",
#         CLASS_IDS['CAP']: "cap"
#     }
#
#     # Initialize annotator
#     annotator = Annotator(
#         img,
#         line_width=VISUALIZATION_PARAMS['line_thickness']
#     )
#
#     # Parse and draw detections
#     if label_path.exists():
#         with open(label_path) as f:
#             for line in f:
#                 parsed = yolo_line_to_xyxy(line.strip(), W, H)
#                 if not parsed:
#                     continue
#
#                 cls_id, box, conf = parsed
#
#                 if conf < conf_threshold:
#                     continue
#
#                 # Prepare label
#                 label = f"{class_names.get(cls_id, 'unknown')} {conf:.2f}"
#
#                 # Draw box
#                 x1, y1, x2, y2 = map(int, box)
#                 annotator.box_label(
#                     [x1, y1, x2, y2],
#                     label,
#                     color=colors(cls_id, True)
#                 )
#
#     # Save result
#     result = annotator.result()
#     cv2.imwrite(str(output_path), result)
#
#     return output_path
#

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

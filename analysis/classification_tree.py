"""
Hierarchical vial state classifier using decision tree architecture.
"""
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np
import cv2
from graphviz import Digraph

from detection.liquid_detector import LiquidDetector
from robotlab_utils.bbox_utils import yolo_line_to_xyxy, box_area
from image_analysis.line_hv_detection import LineDetector
from analysis.gelled_analysis import run_curve_metrics
from config import CLASS_IDS, LIQUID_CLASSES, CURVE_PARAMS, DETECTION_THRESHOLDS, DETECTION_FILTERS, LINE_RULES, \
    LINE_PARAMS, LIQUID_DETECTOR
from robotlab_utils.liquid_detection_utils import parse_detections


class VialState(Enum):
    STABLE = "stable"
    GELLED = "gelled"
    PHASE_SEPARATED = "phase_separated"
    ONLY_AIR = "only_air"
    UNKNOWN = "unknown"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Evidence:
    """Container for all collected evidence from different analyzers."""
    detections: List[Dict[str, Any]]

    # class areas
    liquid_count: int = 0
    gel_count: int = 0
    stable_count: int = 0
    air_count: int = 0
    total_liquid_area: float = 0.0
    gel_area_fraction: float = 0.0

    # geometry (vial ROI)
    image_height: int = 0
    image_width: int = 0
    vial_xL: Optional[int] = None
    vial_xR: Optional[int] = None
    vial_y_top: Optional[int] = 0
    vial_y_bottom: Optional[int] = None
    cap_level_y: Optional[int] = None
    interior_width: Optional[int] = None

    # line detection
    horizontal_lines: Optional[Dict[str, Any]] = None
    vertical_lines: Optional[Dict[str, Any]] = None
    num_horizontal_lines: int = 0
    num_vertical_lines: int = 0
    # horizontal line data
    hline_y: List[int] = field(default_factory=list)
    hline_len_frac: List[float] = field(default_factory=list)
    hline_line_thickness: List[float] = field(default_factory=list)
    hline_x_start: List[float] = field(default_factory=list)
    hline_x_end: List[float] = field(default_factory=list)

    # curve analysis
    curve_stats: Optional[Dict[str, Any]] = None
    curve_variance: float = 0.0
    curve_std_dev: float = 0.0
    curve_roughness: float = 0.0

    # grouping
    merged_liquid_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # xyxy
    image_path: Optional[Path] = None


@dataclass
class Decision:
    """Final decision with confidence and reasoning chain."""
    state: VialState
    confidence: Confidence
    reasoning: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    alternative_states: List[Tuple[VialState, float]] = field(default_factory=list)
    node_path: List[str] = field(default_factory=list)


class DecisionNode:
    """Base class for decision tree nodes."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.children: Dict[str, 'DecisionNode'] = {}

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        """
        Evaluate node and return (next_key, decision).
        If decision is None, traverse to children[next_key].
        If decision is not None, this is a terminal node.
        """
        raise NotImplementedError

    def add_child(self, key: str, node: 'DecisionNode') -> 'DecisionNode':
        self.children[key] = node
        return self


# ============================================================================
# DECISION NODES
# ============================================================================

class RootNode(DecisionNode):
    """Entry point: Check what model detected."""

    def __init__(self):
        super().__init__("root", "Initial model detection analysis")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        # No detections at all
        if len(evidence.detections) == 0:
            return "no_detections", None

        # Only AIR detected
        if evidence.air_count > 0 and evidence.liquid_count == 0:
            return "only_air", None

        # Multiple liquid regions detected
        if evidence.liquid_count >= 2:
            return "multiple_liquids", None

        # Single liquid region
        if evidence.liquid_count == 1:
            return "single_liquid", None

        # Fallback
        return "unknown", None


class NoDetectionsNode(DecisionNode):
    """Handle case where model found nothing."""

    def __init__(self):
        super().__init__("no_detections", "No model detections found")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        # Use image brightness as fallback
        # This would need to be added to Evidence if needed
        reasoning = ["No detections from model", "Checking image characteristics"]

        return None, Decision(
            state=VialState.UNKNOWN,
            confidence=Confidence.LOW,
            reasoning=reasoning,
            evidence_summary={"detections": 0},
            node_path=path.copy()
        )


class OnlyAirNode(DecisionNode):
    """Handle case where only AIR was detected (no liquid)."""

    def __init__(self):
        super().__init__("only_air", "Only AIR detected by model")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Model detected only AIR", "Checking line detection for missed liquid"]

        # Check if multiple horizontal lines exist (missed liquid interface)
        if evidence.num_horizontal_lines >= 2:
            return "air_with_lines", None

        # Check curve analysis
        if evidence.curve_variance >= CURVE_PARAMS.get("gel_variance_thr", 80.0):
            return "air_with_curve", None

        # Check line analysis
        if (evidence.num_horizontal_lines == 1 and
                evidence.curve_variance <= CURVE_PARAMS.get("stable_variance_thr", 50.0) and
                evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75)):
            return "air_with_single_line", None

        # Consider in-between case
        if (CURVE_PARAMS.get("gel_variance_thr", 80.0) > evidence.curve_variance >
                CURVE_PARAMS.get("stable_variance_thr", 50.0)):
            if evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75):
                return "air_with_single_line", None
            else:
                return "air_with_curve", None

        # Truly only air
        reasoning.append("No evidence of liquid in image analysis")
        return None, Decision(
            state=VialState.ONLY_AIR,
            confidence=Confidence.HIGH,
            reasoning=reasoning,
            evidence_summary={
                "air_detections": evidence.air_count,
                "horizontal_lines": evidence.num_horizontal_lines
            },
            node_path=path.copy()
        )


class AirWithLinesNode(DecisionNode):
    """AIR detected but horizontal lines suggest phase separation."""

    def __init__(self):
        super().__init__("air_with_lines", "AIR + horizontal lines detected")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Model detected AIR only",
            f"Found {evidence.num_horizontal_lines} horizontal lines",
            "Lines suggest phase separation missed by model"
        ]

        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "detection_source": "line_analysis",
                "num_lines": evidence.num_horizontal_lines
            },
            alternative_states=[(VialState.ONLY_AIR, 0.3)],
            node_path=path.copy()
        )


class AirWithCurveNode(DecisionNode):
    """AIR detected but curve suggests GEL."""

    def __init__(self):
        super().__init__("air_with_curve", "AIR + high curve variance")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Model detected AIR only",
            f"Curve variance {evidence.curve_variance:.2f} indicates GEL",
            "Gel state missed by model"
        ]

        return None, Decision(
            state=VialState.GELLED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "detection_source": "curve_analysis",
                "curve_variance": evidence.curve_variance
            },
            alternative_states=[(VialState.STABLE, 0.2)],
            node_path=path.copy()
        )

class AirWithSingleLineNode(DecisionNode):
    """AIR detected but single horizontal line suggests STALE (OR possibly GEL)."""

    def __init__(self):
        super().__init__("air_with_single_line", "AIR + single detected horizontal line")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Model detected AIR only",
            f"Detected {evidence.num_horizontal_lines} horizontal line indicates STABLE",
            "STABLE state missed by model"
        ]

        return None, Decision(
            state=VialState.STABLE,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "detection_source": "line_analysis",
                "number of horizontal lines": evidence.num_horizontal_lines,
                "line length fraction": evidence.hline_len_frac,
                "low curve variance": evidence.curve_variance
            },
            alternative_states=[(VialState.GELLED, 0.3)],
            node_path=path.copy()
        )

class MultipleLiquidsNode(DecisionNode):
    """Multiple liquid regions detected by model."""

    def __init__(self):
        super().__init__("multiple_liquids", "Multiple liquid regions detected")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [f"Model detected {evidence.liquid_count} liquid regions"]

        # Check horizontal lines for confirmation
        if evidence.num_horizontal_lines >= 2:
            reasoning.append(f"Confirmed by {evidence.num_horizontal_lines} horizontal lines")
            conf = Confidence.HIGH
        else:
            reasoning.append("No horizontal lines detected for confirmation")
            conf = Confidence.MEDIUM

        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=conf,
            reasoning=reasoning,
            evidence_summary={
                "liquid_regions": evidence.liquid_count,
                "horizontal_lines": evidence.num_horizontal_lines
            },
            node_path=path.copy()
        )


class SingleLiquidNode(DecisionNode):
    """Single liquid region detected - need to classify as gel or stable."""

    def __init__(self):
        super().__init__("single_liquid", "Single liquid region detected")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        # Check model's classification first
        if evidence.gel_count > evidence.stable_count:
            return "model_says_gel", None
        elif evidence.stable_count > evidence.gel_count:
            return "model_says_stable", None
        else:
            # Equal or unclear, go to analysis
            return "unclear_liquid", None


class ModelSaysGelNode(DecisionNode):
    """Model classified as gel - verify with analysis."""

    def __init__(self):
        super().__init__("model_says_gel", "Model classified as GEL")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Model classified as GEL"]

        # Check curve analysis for confirmation
        gel_var_thr = CURVE_PARAMS.get("gel_variance_thr", 80.0)
        stable_var_thr = CURVE_PARAMS.get("stable_variance_thr", 50.0)

        if evidence.curve_variance >= gel_var_thr:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} confirms gel")
            conf = Confidence.HIGH
            alt = []
        elif evidence.curve_variance < stable_var_thr:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} suggests stable")
            reasoning.append("Contradiction between model and curve analysis")
            conf = Confidence.LOW
            alt = [(VialState.STABLE, 0.6)]
        else:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} is ambiguous")
            conf = Confidence.MEDIUM
            alt = [(VialState.STABLE, 0.3)]

        return None, Decision(
            state=VialState.GELLED,
            confidence=conf,
            reasoning=reasoning,
            evidence_summary={
                "gel_area_fraction": evidence.gel_area_fraction,
                "curve_variance": evidence.curve_variance
            },
            alternative_states=alt,
            node_path=path.copy()
        )


class ModelSaysStableNode(DecisionNode):
    """Model classified as stable - verify with analysis."""

    def __init__(self):
        super().__init__("model_says_stable", "Model classified as STABLE")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Model classified as STABLE"]

        # Check curve analysis for contradiction
        gel_var_thr = CURVE_PARAMS.get("gel_variance_thr", 80.0)

        if evidence.curve_variance >= gel_var_thr:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} indicates gel")
            reasoning.append("Overriding model with curve analysis")
            return None, Decision(
                state=VialState.GELLED,
                confidence=Confidence.MEDIUM,
                reasoning=reasoning,
                evidence_summary={
                    "model_classification": "stable",
                    "curve_variance": evidence.curve_variance
                },
                alternative_states=[(VialState.STABLE, 0.3)],
                node_path=path.copy()
            )

        # Check for horizontal line (perfectly stable interface)
        if evidence.num_horizontal_lines == 1:
            reasoning.append("Single horizontal line confirms stable interface")
            conf = Confidence.HIGH
        else:
            reasoning.append("No clear horizontal line detected")
            conf = Confidence.MEDIUM

        return None, Decision(
            state=VialState.STABLE,
            confidence=conf,
            reasoning=reasoning,
            evidence_summary={
                "stable_area_fraction": 1.0 - evidence.gel_area_fraction,
                "horizontal_lines": evidence.num_horizontal_lines,
                "curve_variance": evidence.curve_variance
            },
            node_path=path.copy()
        )


class UnclearLiquidNode(DecisionNode):
    """Model unclear about gel vs stable - rely on analysis."""

    def __init__(self):
        super().__init__("unclear_liquid", "Model classification unclear")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Model classification unclear", "Relying on image analysis"]

        gel_var_thr = CURVE_PARAMS.get("gel_variance_thr", 80.0)
        stable_var_thr = CURVE_PARAMS.get("stable_variance_thr", 50.0)

        # Curve analysis is primary
        if evidence.curve_variance >= gel_var_thr:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} indicates gel")
            state = VialState.GELLED
            conf = Confidence.MEDIUM
        elif evidence.curve_variance < stable_var_thr:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} indicates stable")
            state = VialState.STABLE
            conf = Confidence.MEDIUM
        else:
            reasoning.append(f"Curve variance {evidence.curve_variance:.2f} is ambiguous")
            # Use area-based fallback
            if evidence.gel_area_fraction > 0.5:
                state = VialState.GELLED
                reasoning.append(f"Gel area fraction {evidence.gel_area_fraction:.2f} tips toward gel")
            else:
                state = VialState.STABLE
                reasoning.append(f"Gel area fraction {evidence.gel_area_fraction:.2f} tips toward stable")
            conf = Confidence.LOW

        return None, Decision(
            state=state,
            confidence=conf,
            reasoning=reasoning,
            evidence_summary={
                "curve_variance": evidence.curve_variance,
                "gel_area_fraction": evidence.gel_area_fraction
            },
            node_path=path.copy()
        )


# ============================================================================
# DECISION TREE BUILDER
# ============================================================================
def build_decision_tree() -> DecisionNode:
    """Construct the complete decision tree."""
    root = RootNode()

    # Branch: No detections
    root.add_child("no_detections", NoDetectionsNode())

    # Branch: Only AIR
    air_node = OnlyAirNode()
    air_node.add_child("air_with_lines", AirWithLinesNode())
    air_node.add_child("air_with_curve", AirWithCurveNode())
    air_node.add_child("air_with_single_line", AirWithSingleLineNode())
    root.add_child("only_air", air_node)

    # Branch: Multiple liquids
    root.add_child("multiple_liquids", MultipleLiquidsNode())

    # Branch: Single liquid
    single_node = SingleLiquidNode()
    single_node.add_child("model_says_gel", ModelSaysGelNode())
    single_node.add_child("model_says_stable", ModelSaysStableNode())
    single_node.add_child("unclear_liquid", UnclearLiquidNode())
    root.add_child("single_liquid", single_node)

    # Fallback
    root.add_child("unknown", NoDetectionsNode())

    return root


# ============================================================================
# CLASSIFIER
# ============================================================================

class VialStateClassifierV2:
    """Hierarchical classifier using decision tree."""

    def __init__(self):
        self.tree = build_decision_tree()

    def classify(self,
                 crop_path: Path,
                 label_path: Path,
                 collect_line_analysis: bool = True,
                 collect_curve_analysis: bool = True) -> Dict[str, Any]:
        """
        Classify vial state using hierarchical decision tree.

        Args:
            crop_path: Path to crop image
            label_path: Path to detection labels
            collect_line_analysis: Whether to run line detection
            collect_curve_analysis: Whether to run curve analysis

        Returns:
            Classification result dictionary
        """
        # If label_path not exist or is empty, run liquid detection on the single crop
        if not label_path.exists() or label_path.stat().st_size == 0:
            # Initialize the detector
            detector = LiquidDetector(
                weights_path=LIQUID_DETECTOR.get("liquid_weights", "liquid/best_renamed"),
                task=LIQUID_DETECTOR.get("liquid_task", "detect"),
                img_size=LIQUID_DETECTOR.get("liquid_img_size", 640),
                conf_threshold=LIQUID_DETECTOR.get("liquid_conf", 0.45),
                iou_threshold=LIQUID_DETECTOR.get("liquid_iou", 0.50)
            )

            # Run detection on the single crop image
            temp_out = Path('/tmp/liquid_detect')  # temp output dir
            temp_out.mkdir(parents=True, exist_ok=True)

            results_dir = detector.run_detection(
                source_dir=crop_path,
                output_dir=temp_out,
                save_txt=True,
                save_conf=True,
                save_img=False
            )

            # place generated .txt file to labels dir
            labels_dir = results_dir / 'labels'
            generated_label = labels_dir / f"{crop_path.stem}.txt"

            if generated_label.exists():
                # Copy to expected label_path
                shutil.copy(generated_label, label_path)
            else:
                # No detections found; create an empty label file to proceed
                label_path.touch()

            # Clean up temp dir
            shutil.rmtree(temp_out)

        # Collect all evidence
        evidence = self._collect_evidence(
            crop_path=crop_path,
            label_path=label_path,
            collect_lines=collect_line_analysis,
            collect_curve=collect_curve_analysis
        )

        # Traverse decision tree
        decision = self._traverse_tree(evidence)

        # Format output
        return {
            "vial_state": decision.state.value,
            "confidence": decision.confidence.value,
            "reasoning": decision.reasoning,
            "evidence": decision.evidence_summary,
            "alternative_states": [
                {"state": s.value, "probability": p}
                for s, p in decision.alternative_states
            ],
            "decision_path": " -> ".join(decision.node_path)
        }

    def _collect_evidence(self, crop_path: Path, label_path: Path,
                          collect_lines: bool, collect_curve: bool) -> Evidence:
        """
        Gather all evidence from model.
        """
        img = cv2.imread(str(crop_path))
        H, W = img.shape[:2] if img is not None else (0, 0)

        # Parse detections
        detections = parse_detections(label_path, W, H)

        # parse & filter detections
        # detections = []
        # if label_path.exists():
        #     with open(label_path) as f:
        #         for line in f:
        #             parsed = yolo_line_to_xyxy(line.strip(), W, H)
        #             if not parsed: continue
        #             cls_id, box, conf = parsed
        #             if conf < DETECTION_FILTERS["conf_min"]: continue
        #             area = box_area(box)
        #             if area < DETECTION_FILTERS["min_liquid_area_frac"] * (W * H) and cls_id in LIQUID_CLASSES:
        #                 continue
        #             detections.append({'class_id': cls_id, 'box': box, 'confidence': conf, 'area': area})

        # counts & areas
        liquid_count = sum(1 for d in detections if d['class_id'] in LIQUID_CLASSES)
        gel_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['GEL'])
        stable_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['STABLE'])
        air_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['AIR'])
        total_liquid_area = sum(d['area'] for d in detections if d['class_id'] in LIQUID_CLASSES)
        gel_area = sum(d['area'] for d in detections if d['class_id'] == CLASS_IDS['GEL'])
        gel_area_fraction = gel_area / total_liquid_area if total_liquid_area > 0 else 0.0

        ev = Evidence(
            detections=detections,
            liquid_count=liquid_count, gel_count=gel_count, stable_count=stable_count, air_count=air_count,
            total_liquid_area=total_liquid_area, gel_area_fraction=gel_area_fraction,
            image_height=H, image_width=W, vial_y_bottom=H, image_path=crop_path
        )

        # lines
        if collect_lines and crop_path.exists():
            # init line detector with config params
            det = LineDetector(
                horiz_kernel_div=LINE_PARAMS.get("horiz_kernel_div", 15),
                vert_kernel_div=LINE_PARAMS.get("vert_kernel_div", 30),
                adaptive_block=LINE_PARAMS.get("adaptive_block", 15),
                adaptive_c=LINE_PARAMS.get("adaptive_c", -2),
                min_line_length=LINE_PARAMS.get("min_line_length", 0.3),
                min_line_strength=LINE_PARAMS.get("min_line_strength", 0.1),
                merge_threshold=LINE_PARAMS.get("merge_threshold", 0.1),
            )
            result = det.detect(crop_path, bottom_exclusion=LINE_PARAMS.get("bottom_exclusion", 0.15),
                                top_exclusion=LINE_PARAMS.get("top_exclusion", 0.30))
            ev.horizontal_lines = result.get('horizontal_lines', {})
            ev.vertical_lines = result.get('vertical_lines', {})
            ev.num_horizontal_lines = ev.horizontal_lines.get('num_lines', 0)
            ev.num_vertical_lines = ev.vertical_lines.get('num_lines', 0)

            # vial walls from verticals
            xL = ev.vertical_lines.get('x_left')
            xR = ev.vertical_lines.get('x_right')
            if xL is not None and xR is not None and xR > xL:
                ev.vial_xL, ev.vial_xR = int(xL), int(xR)
                ev.interior_width = int(xR - xL)
                ev.vial_y_top = 0
                cap_frac = LINE_RULES["cap_level_frac"]
                ev.cap_level_y = int(cap_frac * H)

            # horizontal line features
            lines = ev.horizontal_lines.get('lines', [])
            for L in lines:
                ev.hline_y.append(L.get('y_position', 0))
                ev.hline_len_frac.append(L.get('x_length_frac', 0.0))
                ev.hline_x_start.append(L.get('x_start', 0.0))
                ev.hline_x_end.append(L.get('x_end', 0.0))
                ev.hline_line_thickness.append(L.get('thickness', 0.0))

        # curve
        if collect_curve and crop_path.exists():
            c = run_curve_metrics(crop_path) or {}
            stats = c.get('stats', {}) or {}
            ev.curve_stats = stats
            ev.curve_variance = stats.get('variance') or stats.get('variance_from_baseline') or 0.0
            ev.curve_std_dev = stats.get('std') or stats.get('std_dev_from_baseline') or 0.0
            ev.curve_roughness = stats.get('roughness', 0.0)

        # merge liquid fragments
        liquid_xyxy = [d['box'] for d in detections if d['class_id'] in LIQUID_CLASSES]
        ev.liquid_count = len(liquid_xyxy)

        return ev

    def _traverse_tree(self, evidence: Evidence) -> Decision:
        """Traverse decision tree until reaching terminal node."""
        current_node = self.tree
        path = []
        max_depth = 20  # Prevent infinite loops
        depth = 0

        while depth < max_depth:
            next_key, decision = current_node.evaluate(evidence, path)

            # Terminal node reached
            if decision is not None:
                return decision

            # Move to next node
            if next_key and next_key in current_node.children:
                current_node = current_node.children[next_key]
                depth += 1
            else:
                # No valid path, return unknown
                return Decision(
                    state=VialState.UNKNOWN,
                    confidence=Confidence.LOW,
                    reasoning=[f"Tree traversal failed at {current_node.name}"],
                    node_path=path
                )

        # Max depth exceeded
        return Decision(
            state=VialState.UNKNOWN,
            confidence=Confidence.LOW,
            reasoning=["Max tree depth exceeded"],
            node_path=path
        )


def export_tree_graphviz(
    root: DecisionNode,
    out_dot: Union[Path, str],
    out_png: Optional[Union[Path, str]] = None,
    title: str = "Vial State Decision Tree",
    highlight_path: Optional[List[str]] = None,
) -> Digraph:
    """
    Export the DecisionNode tree as Graphviz DOT/PNG.

    highlight_path: list of node names visited in order (e.g., decision.node_path).
                    When provided, those nodes and their connecting edges are highlighted.
    """
    dot = Digraph(name="VialStateTree", format="png")
    dot.attr(label=title, labelloc="t", fontsize="18")
    dot.attr("graph", rankdir="TB", splines="spline", nodesep="0.35", ranksep="0.4")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#f7f7f9",
             color="#999999", fontname="Helvetica", fontsize="10")
    dot.attr("edge", color="#666666", fontname="Helvetica", fontsize="9")

    id_map: dict[int, str] = {}
    counter = {"n": 0}
    highlight_nodes = set(highlight_path or [])

    # build parent->child-name->key map for edge highlighting
    edge_in_path: set[tuple[str, str]] = set()
    if highlight_path and len(highlight_path) >= 2:
        # walk down root using names
        name_to_nodes: dict[str, list[DecisionNode]] = {}

        def index_nodes(n: DecisionNode):
            name_to_nodes.setdefault(n.name, []).append(n)
            for ch in n.children.values():
                index_nodes(ch)
        index_nodes(root)

        # reconstruct pairs by searching the unique child for each hop
        for i in range(len(highlight_path) - 1):
            p_name = highlight_path[i]
            c_name = highlight_path[i + 1]
            # disambiguate if names repeat: pick any parent instance that has a child with that name
            for p in name_to_nodes.get(p_name, []):
                for key, ch in p.children.items():
                    if ch.name == c_name:
                        edge_in_path.add((p_name, c_name))
                        break

    def node_id(n: DecisionNode) -> str:
        if id(n) not in id_map:
            counter["n"] += 1
            id_map[id(n)] = f"N{counter['n']}"
        return id_map[id(n)]

    def is_terminal(n: DecisionNode) -> bool:
        return len(n.children) == 0

    def add_node(n: DecisionNode):
        nid = node_id(n)

        # base style depending on terminal status
        if is_terminal(n):
            attrs = dict(shape="doubleoctagon", fillcolor="#eef7ff")
        else:
            attrs = dict(shape="box", fillcolor="#f7f7f9")

        # highlight if node is on decision path
        if n.name in highlight_nodes:
            attrs.update({
                "color": "#FF6A00",
                "penwidth": "2.6",
                "fillcolor": "#FFE6CC"
            })

        dot.node(nid, f"{n.name}", **attrs)

    def walk(n: DecisionNode, seen: set[int]):
        add_node(n)
        seen.add(id(n))
        for key, child in n.children.items():
            add_node(child)
            # style for edges on highlight path
            if (n.name, child.name) in edge_in_path:
                dot.edge(node_id(n), node_id(child), label=key, color="#FF6A00", penwidth="2.6")
            else:
                dot.edge(node_id(n), node_id(child), label=key)
            if id(child) not in seen:
                walk(child, seen)

    walk(root, set())

    out_dot = Path(out_dot)
    out_dot.parent.mkdir(parents=True, exist_ok=True)
    dot.save(filename=str(out_dot))  # writes .dot

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        dot.render(filename=str(out_png.with_suffix("")), format="png", cleanup=True)

    return dot

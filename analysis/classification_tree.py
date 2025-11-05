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
from robotlab_utils.bbox_utils import yolo_line_to_xyxy, box_area, expand_and_clamp, ensure_xyxy_px
from image_analysis.line_hv_detection import LineDetector
from analysis.gelled_analysis import run_curve_metrics
from config import CLASS_IDS, LIQUID_CLASSES, CURVE_PARAMS, DETECTION_THRESHOLDS, DETECTION_FILTERS, LINE_RULES, \
    LINE_PARAMS, LIQUID_DETECTOR, REGION_RULES
from robotlab_utils.image_utils import extract_region, calculate_brightness_stats
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

        # Single liquid region
        if evidence.liquid_count == 1:
            return "single_liquid", None

        # Multiple liquid regions
        if evidence.liquid_count >= 2:
            return "multiple_liquids", None

        # Fallback
        return None, Decision (
            state=VialState.UNKNOWN,
            confidence=Confidence.LOW,
            reasoning=["Fallback: Unexpected detection pattern"],
            node_path=path
        )


class NoDetectionsNode(DecisionNode):
    """Handle case with no detections."""
    def __init__(self):
        super().__init__("no_detections", "No model detections")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["No detections from model"]

        # Potential overrides
        if evidence.curve_variance >= CURVE_PARAMS["gel_variance_thr"]:
            reasoning.append("Curve analysis override to gelled")
            return None, Decision(
                state=VialState.GELLED,
                confidence=Confidence.MEDIUM,
                reasoning=reasoning,
                evidence_summary={"curve_variance": evidence.curve_variance},
                alternative_states=[(VialState.ONLY_AIR, 0.3)],
                node_path=path
            )

        if evidence.num_horizontal_lines >= 2:
            reasoning.append("Multiple lines suggest phase separation")
            return None, Decision(
                state=VialState.PHASE_SEPARATED,
                confidence=Confidence.MEDIUM,
                reasoning=reasoning,
                evidence_summary={"num_horizontal_lines": evidence.num_horizontal_lines},
                alternative_states=[(VialState.STABLE, 0.3)],
                node_path=path
            )

        if evidence.num_horizontal_lines == 1 and evidence.curve_variance <= CURVE_PARAMS["stable_variance_thr"]:
            reasoning.append("Single horizontal line suggests stable")
            return None, Decision(
                state=VialState.STABLE,
                confidence=Confidence.MEDIUM,
                reasoning=reasoning,
                evidence_summary={"num_horizontal_lines": evidence.num_horizontal_lines},
                alternative_states=[(VialState.ONLY_AIR, 0.3)],
                node_path=path
            )

        # Fallback to unknown or only air
        return None, Decision(
            state=VialState.UNKNOWN,
            confidence=Confidence.LOW,
            reasoning=reasoning + ["No overrides from analysis"],
            alternative_states=[(VialState.ONLY_AIR, 0.4)],
            node_path=path
        )


class OnlyAirNode(DecisionNode):
    """Handle case where only AIR is detected."""
    def __init__(self):
        super().__init__("only_air", "Only AIR detected")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Model detected only AIR", "Checking image analysis for missed liquid"]

        # Multiple lines override
        if evidence.num_horizontal_lines >= 2:
            return "air_with_lines", None

        # Curve override for gel
        if evidence.curve_variance >= CURVE_PARAMS.get("gel_variance_thr", 75.0):
            return "air_with_curve", None

        # Single line with low curve variance
        if evidence.num_horizontal_lines == 1 and evidence.curve_variance <= CURVE_PARAMS.get("stable_variance_thr", 55.0):
            if len(evidence.hline_len_frac) > 0 and evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75):
                return "air_with_single_line", None

        # Consider in-between case
        if (CURVE_PARAMS.get("gel_variance_thr", 75.0) > evidence.curve_variance >
                CURVE_PARAMS.get("stable_variance_thr", 55.0)):
            if evidence.num_horizontal_lines > 0 and len(evidence.hline_len_frac) > 0 and evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75):
                return "air_with_single_line", None
            else:
                return "air_with_curve", None

        # Default to only air
        reasoning.append("No overrides from image analysis")
        return None, Decision(
            state=VialState.ONLY_AIR,
            confidence=Confidence.HIGH,
            reasoning=reasoning,
            evidence_summary={
                "air_count": evidence.air_count,
                "num_horizontal_lines": evidence.num_horizontal_lines,
                "curve_variance": evidence.curve_variance
            },
            node_path=path
        )


class AirWithLinesNode(DecisionNode):
    """Air with multiple horizontal lines."""
    def __init__(self):
        super().__init__("air_with_lines", "Air with multiple lines")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Only AIR detected",
            f"Found {evidence.num_horizontal_lines} horizontal lines",
            "Lines suggest phase separation missed by model"
        ]

        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "num_horizontal_lines": evidence.num_horizontal_lines
            },
            alternative_states=[(VialState.ONLY_AIR, 0.3)],
            node_path=path
        )


class AirWithCurveNode(DecisionNode):
    """AIR detected but curve suggests GEL."""
    def __init__(self):
        super().__init__("air_with_curve", "Air with curve override")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Only air detected",
            f"High curve variance ({evidence.curve_variance:.2f}) suggests gelled state"
        ]
        return None, Decision(
            state=VialState.GELLED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "detection_source": "curve_analysis",
                "curve_variance": evidence.curve_variance
            },
            alternative_states=[(VialState.STABLE, 0.3)],
            node_path=path
        )


class AirWithSingleLineNode(DecisionNode):
    """Air with single strong horizontal line."""
    def __init__(self):
        super().__init__("air_with_single_line", "Air with single line override")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Only AIR detected",
            "Single strong horizontal line and low curve variance suggest stable liquid"
        ]
        return None, Decision(
            state=VialState.STABLE,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "num_horizontal_lines": evidence.num_horizontal_lines,
                "line_length_fraction": evidence.hline_len_frac[0] if evidence.hline_len_frac else 0,
                "curve_variance": evidence.curve_variance
            },
            alternative_states=[(VialState.GELLED, 0.3)],
            node_path=path
        )


class MultipleLiquidsNode(DecisionNode):
    """Handle multiple liquid detections."""
    def __init__(self):
        super().__init__("multiple_liquids", "Multiple liquid regions")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        # print("\nAIRCOUNT\n", evidence.air_count)
        if evidence.air_count > 0:
            return "multiple_with_air", None
        else:
            return "multiple_no_air", None

        # reasoning = [f"Multiple liquid regions detected ({evidence.liquid_count})"]
        #
        # # Confidence boost with lines
        # conf = Confidence.HIGH if evidence.num_horizontal_lines >= 2 else Confidence.MEDIUM
        # if evidence.num_horizontal_lines >= 2:
        #     reasoning.append("Confirmed by multiple horizontal lines")
        #
        # return None, Decision(
        #     state=VialState.PHASE_SEPARATED,
        #     confidence=conf,
        #     reasoning=reasoning,
        #     evidence_summary={
        #         "liquid_count": evidence.liquid_count,
        #         "num_horizontal_lines": evidence.num_horizontal_lines
        #     },
        #     node_path=path
        # )


class MultipleWithAirNode(DecisionNode):
    """Multiple liquid regions with AIR present."""
    def __init__(self):
        super().__init__("multiple_with_air", "Multiple liquids + air")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        reasoning = ["Multiple liquid regions detected with AIR present"]
        if evidence.num_horizontal_lines >= 2:
            reasoning.append("Confirmed by multiple horizontal lines")
            conf = Confidence.HIGH
        else:
            reasoning.append("Only one/no line; still multiple regions present")
            conf = Confidence.MEDIUM

        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=conf,
            reasoning=reasoning,
            evidence_summary={
                "liquid_count": evidence.liquid_count,
                "air_count": evidence.air_count,
                "num_horizontal_lines": evidence.num_horizontal_lines
            },
            node_path=path
        )


class MultipleNoAirNode(DecisionNode):
    """
    Multiple liquid regions but no AIR detected.
    If the top-most liquid 'touches top' and either spans deep or we only see 1 horizontal line,
    reclassify that top box to AIR and route to the single_liquid branch.
    Otherwise, if ≥2 lines and the top-most liquid does NOT touch top → PS.
    """
    def __init__(self):
        super().__init__("multiple_no_air", "Multiple liquids, no air")

    @staticmethod
    def _recompute(ev: Evidence) -> None:
        total_liquid_area = sum(d['area'] for d in ev.detections if d['class_id'] in LIQUID_CLASSES)
        gel_area = sum(d['area'] for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.total_liquid_area = total_liquid_area
        ev.gel_area_fraction = gel_area / total_liquid_area if total_liquid_area > 0 else 0.0
        ev.gel_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.stable_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['STABLE'])
        ev.air_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['AIR'])
        ev.liquid_count = ev.gel_count + ev.stable_count

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        H = evidence.image_height

        # Find top-most liquid box
        liquid_dets = [d for d in evidence.detections if d['class_id'] in LIQUID_CLASSES]
        if not liquid_dets:
            # Safety fallback
            return None, Decision(
                state=VialState.UNKNOWN, confidence=Confidence.LOW,
                reasoning=["Multiple liquids expected but none present after filtering"],
                node_path=path
            )

        top_liq = min(liquid_dets, key=lambda d: d['box'][1])
        y1, y2 = top_liq['box'][1], top_liq['box'][3]

        # Thresholds
        touch_thr = REGION_RULES.get("air_top_touch_frac", 0.25) * H
        deep_thr  = REGION_RULES.get("air_deep_span_frac", 0.75) * H

        # Measured cap level
        cap_ok = (evidence.cap_level_y is not None) and (y1 <= evidence.cap_level_y)
        near_top = (y1 <= touch_thr) or cap_ok
        spans_deep = (y2 >= deep_thr)

        # Choose longest line if present for trimming
        interface_y = None
        if evidence.hline_y and evidence.hline_len_frac:
            idx = int(np.argmax(evidence.hline_len_frac))
            interface_y = evidence.hline_y[idx]

        # how much space left on top
        headspace_frac = y1 / float(H)
        min_headspace = REGION_RULES.get("min_headspace_frac", 0.10)
        has_headspace = headspace_frac >= min_headspace
        print("\nHEADSPACE\n", headspace_frac)
        print("\nHEADSPACE\n", min_headspace)
        print("\nHEADSPACE\n", has_headspace)

        # Case 1: likely AIR FP (top-most liquid actually AIR)
        if near_top and not has_headspace and (evidence.num_horizontal_lines == 1 or spans_deep):
            # Reclassify top-most liquid as AIR and trim to interface
            top_liq['class_id'] = CLASS_IDS['AIR']
            top_liq['reclassified_from'] = 'LIQUID→AIR (near_top + 1 line or deep span)'
            if interface_y is not None and top_liq['box'][3] > interface_y:
                top_liq['box'][3] = interface_y  # keep AIR above the interface

            # Recompute counts/areas
            self._recompute(evidence)

            # Route to single_liquid path
            return "route_single", None

        # Case 2: clear PS
        if evidence.num_horizontal_lines >= 2 and not near_top:
            return None, Decision(
                state=VialState.PHASE_SEPARATED,
                confidence=Confidence.HIGH,
                reasoning=[
                    "Multiple liquids with no AIR",
                    "Top-most liquid does not touch the top",
                    f"{evidence.num_horizontal_lines} horizontal lines confirm phase separation"
                ],
                evidence_summary={
                    "liquid_count": evidence.liquid_count,
                    "num_horizontal_lines": evidence.num_horizontal_lines
                },
                node_path=path
            )

        # Default: multiple liquids (no AIR), ambiguous lines - PS
        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=Confidence.MEDIUM,
            reasoning=[
                "Multiple liquids with no AIR",
                "Ambiguous line evidence (not reclassified to AIR)"
            ],
            evidence_summary={
                "liquid_count": evidence.liquid_count,
                "num_horizontal_lines": evidence.num_horizontal_lines
            },
            node_path=path
        )


class SingleLiquidNode(DecisionNode):
    """Handle single liquid detection."""
    def __init__(self):
        super().__init__("single_liquid", "Single liquid region")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        if evidence.air_count > 0:
            return "single_with_air", None
        else:
            return "single_no_air", None


class SingleWithAirNode(DecisionNode):
    """Single liquid with air."""
    def __init__(self):
        super().__init__("single_with_air", "Single liquid with air")

    def _clamp_xyxy(self, box, W, H):
        x1, y1, x2, y2 = map(float, box)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        x1 = max(0.0, min(W - 1.0, x1))
        y1 = max(0.0, min(H - 1.0, y1))
        x2 = max(0.0, min(W - 1.0, x2))
        y2 = max(0.0, min(H - 1.0, y2))
        return [x1, y1, x2, y2]

    def _primary_interface_y(self, evidence):
        # prefer the longest horizontal line
        hl = (evidence.horizontal_lines or {}).get("lines") or []
        if not hl:
            return None
        # lines may already carry x_length_frac; fall back to thickness if needed
        lens = [l.get("x_length_frac", 0.0) for l in hl]
        idx = int(np.argmax(lens)) if lens else 0
        y = hl[idx]["y_position"]
        if (evidence.horizontal_lines or {}).get("y_normalized", False):
            return float(y) * float(evidence.image_height)
        return float(y)

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)

        H = max(1, int(evidence.image_height))
        W = max(1, int(evidence.image_width))
        vert_gap_frac = REGION_RULES.get("vertical_gap_frac", 0.12)  # tolerate small bbox slack
        touch_eps_px  = REGION_RULES.get("vertical_touch_eps_px", max(3, int(0.005 * H)))  # ~0.5% H

        air_dets = [d for d in evidence.detections if d['class_id'] == CLASS_IDS['AIR']]
        liquid_dets = [d for d in evidence.detections if d['class_id'] in LIQUID_CLASSES]

        if air_dets and liquid_dets:
            # top-most AIR by y1
            air_det = min(air_dets, key=lambda d: d['box'][1])
            air_box = self._clamp_xyxy(air_det['box'], W, H)

            # choose the liquid that is directly below AIR (smallest y1 >= air_bottom - eps)
            air_bottom = air_box[3]
            cand = sorted(liquid_dets, key=lambda d: d['box'][1])
            liq_det = None
            for d in cand:
                box = self._clamp_xyxy(d['box'], W, H)
                if box[1] >= air_bottom - touch_eps_px:
                    liq_det = d
                    liq_box = box
                    break
            # if none found (all start above air_bottom due to jitter), take top-most liquid
            if liq_det is None:
                liq_det = cand[0]
                liq_box = self._clamp_xyxy(liq_det['box'], W, H)

            liq_top = liq_box[1]

            # snap to measured interface if available
            interface_y = self._primary_interface_y(evidence)
            if interface_y is not None:
                # if interface lies between (or very near) the two edges, treat as touching
                if (air_bottom - touch_eps_px) <= interface_y <= (liq_top + touch_eps_px):
                    air_bottom = interface_y
                    liq_top    = interface_y

            raw_gap = liq_top - air_bottom
            gap_px  = max(0.0, raw_gap - touch_eps_px)
            gap_frac = gap_px / H

            # if getattr(evidence, "debug", False):
            print("\nAIR COORDS (clamped):", air_box)
            print("LIQ COORDS (clamped):", liq_box)
            print("interface_y:", interface_y, "touch_eps_px:", touch_eps_px)
            print("RAW GAP / GAP PX / FRAC:", raw_gap, gap_px, gap_frac)

            # require ≥2 interfaces to call phase separation from a vertical-gap heuristic
            has_two_interfaces = evidence.num_horizontal_lines >= 2
            if gap_frac >= vert_gap_frac and has_two_interfaces:
                conf = Confidence.HIGH
                reasoning = [
                    "Single liquid + AIR detected",
                    f"Vertical gap = {gap_frac*100:.1f}% (≥ {vert_gap_frac*100:.1f}%)",
                    "≥2 horizontal interfaces detected",
                    "Consistent with phase separation"
                ]
                return None, Decision(
                    state=VialState.PHASE_SEPARATED,
                    confidence=conf,
                    reasoning=reasoning,
                    evidence_summary={
                        "gap_frac": round(gap_frac, 4),
                        "gap_px": int(gap_px),
                        "num_horizontal_lines": evidence.num_horizontal_lines
                    },
                    node_path=path.copy()
                )

            # otherwise prefer stable when only one interface is present
            return "stable_dominant", None

        # no AIR or no LIQUID - delegate majority vote
        if evidence.gel_count > evidence.stable_count:
            return "gel_dominant", None
        elif evidence.stable_count > evidence.gel_count:
            return "stable_dominant", None
        else:
            return "mixed_liquid", None


class GelDominantNode(DecisionNode):
    """Gel dominant in liquid."""
    def __init__(self):
        super().__init__("gel_dominant", "Gel dominant")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Single liquid with air", "Gel detections dominant"]
        return None, Decision(
            state=VialState.GELLED,
            confidence=Confidence.HIGH,
            reasoning=reasoning,
            evidence_summary={
                "gel_count": evidence.gel_count,
                "stable_count": evidence.stable_count
            },
            node_path=path
        )


class StableDominantNode(DecisionNode):
    """Stable dominant in liquid."""
    def __init__(self):
        super().__init__("stable_dominant", "Stable dominant")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Single liquid with air", "Stable detections dominant"]
        return None, Decision(
            state=VialState.STABLE,
            confidence=Confidence.HIGH,
            reasoning=reasoning,
            evidence_summary={
                "stable_count": evidence.stable_count,
                "gel_count": evidence.gel_count
            },
            node_path=path
        )


class MixedLiquidNode(DecisionNode):
    """Mixed gel/stable detections."""
    def __init__(self):
        super().__init__("mixed_liquid", "Mixed liquid types")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Single liquid with air", "Mixed gel and stable detections"]

        state = VialState.GELLED if evidence.gel_area_fraction > 0.5 else VialState.STABLE
        reasoning.append(f"Dominant by area: {state.value} (gel fraction {evidence.gel_area_fraction:.2f})")

        return None, Decision(
            state=state,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={"gel_area_fraction": evidence.gel_area_fraction},
            alternative_states=[(VialState.UNKNOWN, 0.2)],
            node_path=path
        )


class SingleNoAirNode(DecisionNode):
    """Single liquid without air."""
    def __init__(self):
        super().__init__("single_no_air", "Single liquid no air")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Single liquid detected", "No air detections"]

        # Multiple lines override
        if evidence.num_horizontal_lines >= 2:
            return "no_air_with_lines", None

        # Curve override
        if evidence.curve_variance >= CURVE_PARAMS.get("gel_variance_thr", 80.0):
            return "no_air_with_curve", None

        # Single line with low variance
        if evidence.num_horizontal_lines == 1 and evidence.curve_variance <= CURVE_PARAMS.get("stable_variance_thr", 50.0):
            if len(evidence.hline_len_frac) > 0 and evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75):
                return "no_air_with_single_line", None

        # In-between case
        if (CURVE_PARAMS.get("gel_variance_thr", 80.0) > evidence.curve_variance >
                CURVE_PARAMS.get("stable_variance_thr", 50.0)):
            if len(evidence.hline_len_frac) > 0 and evidence.hline_len_frac[0] > LINE_PARAMS.get("min_line_length", 0.75):
                return "no_air_with_single_line", None
            else:
                return "no_air_with_curve", None

        # No indicators
        return "no_indicators", None


class NoAirWithLinesNode(DecisionNode):
    """No air with multiple lines."""
    def __init__(self):
        super().__init__("no_air_with_lines", "No air with multiple lines")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Single liquid no air",
            f"Multiple horizontal lines ({evidence.num_horizontal_lines}) suggest phase separation"
        ]
        return None, Decision(
            state=VialState.PHASE_SEPARATED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={"num_horizontal_lines": evidence.num_horizontal_lines},
            alternative_states=[(VialState.STABLE, 0.2)],
            node_path=path
        )


class NoAirWithCurveNode(DecisionNode):
    """No air with high curve variance."""
    def __init__(self):
        super().__init__("no_air_with_curve", "No air with curve")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Single liquid no air",
            f"High curve variance ({evidence.curve_variance:.2f}) suggests gelled"
        ]
        return None, Decision(
            state=VialState.GELLED,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={"curve_variance": evidence.curve_variance},
            alternative_states=[(VialState.STABLE, 0.3)],
            node_path=path
        )


class NoAirWithSingleLineNode(DecisionNode):
    """No air with single line."""
    def __init__(self):
        super().__init__("no_air_with_single_line", "No air with single line")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = [
            "Single liquid no air",
            "Single strong line and low variance suggest stable"
        ]
        return None, Decision(
            state=VialState.STABLE,
            confidence=Confidence.MEDIUM,
            reasoning=reasoning,
            evidence_summary={
                "num_horizontal_lines": 1,
                "line_length_fraction": evidence.hline_len_frac[0] if evidence.hline_len_frac else 0,
                "curve_variance": evidence.curve_variance
            },
            alternative_states=[(VialState.GELLED, 0.3)],
            node_path=path
        )


class NoAirNoIndicatorsNode(DecisionNode):
    """No air and no overrides."""
    def __init__(self):
        super().__init__("no_indicators", "No indicators for override")

    def evaluate(self, evidence: Evidence, path: List[str]) -> Tuple[Optional[str], Optional[Decision]]:
        path.append(self.name)
        reasoning = ["Single liquid no air", "No line or curve indicators"]
        return None, Decision(
            state=VialState.UNKNOWN,
            confidence=Confidence.LOW,
            reasoning=reasoning,
            evidence_summary={
                "num_horizontal_lines": evidence.num_horizontal_lines,
                "curve_variance": evidence.curve_variance
            },
            alternative_states=[(VialState.STABLE, 0.4), (VialState.GELLED, 0.3)],
            node_path=path
        )


def _apply_air_placement_rules(ev: Evidence) -> None:
    """
    Enforce: AIR must be the single top-most region above the interface.
    Also fixes two common failure modes:
      (1) Top liquid FP that actually represents AIR.
      (2) AIR FN when there is empty space above the highest detection.
    """
    H, W = ev.image_height, ev.image_width
    if H == 0 or W == 0 or not ev.detections:
        return

    cap_y = int(LINE_RULES.get("cap_level_frac", 0.10) * H)

    # choose longest horizontal line
    interface_y = None
    if ev.hline_y:
        # print("\n interface_y REACHED \n")
        try:
            idx = int(np.argmax(ev.hline_len_frac))
        except Exception:
            idx = 0
        interface_y = ev.hline_y[idx]

    tol_px = max(2, int(0.01 * H))  # small tolerance on line position

    def is_topmost(det):
        return det['box'][1] == min(d['box'][1] for d in ev.detections)

    def recompute_counts():
        ev.liquid_count = sum(1 for d in ev.detections if d['class_id'] in LIQUID_CLASSES)
        ev.gel_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.stable_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['STABLE'])
        ev.air_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['AIR'])
        ev.total_liquid_area = sum(d['area'] for d in ev.detections if d['class_id'] in LIQUID_CLASSES)
        gel_area = sum(d['area'] for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.gel_area_fraction = (gel_area / ev.total_liquid_area) if ev.total_liquid_area > 0 else 0.0

    def most_likely_liquid_class():
        return CLASS_IDS['GEL'] if ev.gel_count >= ev.stable_count else CLASS_IDS['STABLE']

    recompute_counts()

    # if AIR is present: must be single, top-most, above interface
    air_dets = [d for d in ev.detections if d['class_id'] == CLASS_IDS['AIR']]
    if len(air_dets) > 0:
        print("\nAIR Present REACHED\n")
        # # if multiple AIR, keep only top-most AIR region
        # if len(air_dets) > 1:
        #     top_air = min(air_dets, key=lambda d: d['box'][1])
        #     for d in air_dets:
        #         if d is top_air:
        #             continue
        #         d['class_id'] = most_likely_liquid_class()
        #     recompute_counts()
        #     air_dets = [top_air]

        # AIR deduplication
        if len(air_dets) > 1:
            # discard small ones (area fraction < 1% of image)
            min_area = 0.05 * (ev.image_width * ev.image_height)
            air_valid = [d for d in air_dets if d['area'] >= min_area]

            # discard those entirely in bottom of image
            h_thr = 0.5 * ev.image_height
            air_valid = [d for d in air_valid if d['box'][1] < h_thr]

            if air_valid:
                # pick the one with largest area
                best_air = max(air_valid, key=lambda d: d['area'])
                for d in air_dets:
                    if d is not best_air:
                        d['class_id'] = most_likely_liquid_class()
                ev.air_count = 1
            else:
                # fallback: keep top-most if everything was filtered
                top_air = min(air_dets, key=lambda d: d['box'][1])
                for d in air_dets:
                    if d is not top_air:
                        d['class_id'] = most_likely_liquid_class()
                ev.air_count = 1

        air = air_dets[0]
        y1, y2 = air['box'][1], air['box'][3]

        if not is_topmost(air):
            print("\nAIR NOT TOPMOST REACHED\n")
            # AIR below a liquid region - reclassify
            air['class_id'] = most_likely_liquid_class()
            recompute_counts()
        else:
            print("\nAIR TOPMOST REACHED\n")
            # required to be above the interface and start near top
            above_interface_ok = True
            if interface_y is not None:
                above_interface_ok = (y2 <= interface_y + tol_px)

            if not above_interface_ok or y1 > cap_y:
                # AIR bottom below interface or AIR not start near top - reclassify
                air['class_id'] = most_likely_liquid_class()
                recompute_counts()

    # If AIR is missing: infer AIR at the top when obvious
    if sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['AIR']) == 0:
        print("\nAIR INFER AT TOP REACHED\n")
        sorted_dets = sorted(ev.detections, key=lambda d: d['box'][1])
        top_det = sorted_dets[0]
        ty1, ty2 = top_det['box'][1], top_det['box'][3]

        # Empty space above highest detection - synthesize AIR
        empty_space_frac_thr = max(0.12, 2.0 * LINE_RULES.get("cap_level_frac", 0.08))
        gap_frac = ty1 / float(H)
        empty_space_condition = gap_frac >= empty_space_frac_thr
        if interface_y is not None:
            empty_space_condition = empty_space_condition or (ty1 >= interface_y - tol_px)
        if empty_space_condition:
            # reclassify the top-most detection

            print("\nREACHED\n")

            top_det['class_id'] = CLASS_IDS['AIR']
            top_det['source'] = 'air_reclassified_from_liquid'
            # Adjust bottom boundary to match interface
            if interface_y is not None:
                top_det['box'][3] = min(top_det['box'][3], interface_y - tol_px)
            else:
                top_det['box'][3] = ty1  # keep above previous liquid start
            # Recalculate area
            top_det['area'] = float((top_det['box'][3] - top_det['box'][1]) * W)
            recompute_counts()

        else:
            # The top-most detection touches top - likely AIR false positive
            if ty1 <= cap_y and len(sorted_dets) >= 2:
                top_det['class_id'] = CLASS_IDS['AIR']
                if interface_y is not None:
                    top_det['box'][3] = min(top_det['box'][3], interface_y)
                recompute_counts()

    # brightness tie-breaker if still no AIR after geometry
    if sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['AIR']) == 0:
        if ev.image_path and ev.image_path.exists():
            img = cv2.imread(str(ev.image_path))
            if img is not None and ev.detections:
                # top-most region
                top_det = min(ev.detections, key=lambda d: d['box'][1])
                # only if it is liquid class
                if top_det['class_id'] in LIQUID_CLASSES:
                    x1, y1, x2, y2 = map(int, top_det['box'])
                    roi = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if roi.size > 0:
                        mean = float(roi.mean())
                        thr = DETECTION_THRESHOLDS.get('air_brightness_thr', 80.0)
                        if mean < thr:
                            top_det['class_id'] = CLASS_IDS['AIR']
                            top_det['source'] = 'air_reclassified_by_brightness'
                            # trim to interface if present
                            if 'hline_y' in ev.__dict__ and ev.hline_y:
                                idx = int(np.argmax(ev.hline_len_frac))
                                interface_y = ev.hline_y[idx]
                                top_det['box'][3] = min(top_det['box'][3], interface_y)


def _build_tree() -> DecisionNode:
    """Build the decision tree structure."""
    root = RootNode()

    # No detections branch
    root.add_child("no_detections", NoDetectionsNode())

    # Only air branch
    only_air = OnlyAirNode()
    only_air.add_child("air_with_lines", AirWithLinesNode())
    only_air.add_child("air_with_curve", AirWithCurveNode())
    only_air.add_child("air_with_single_line", AirWithSingleLineNode())
    root.add_child("only_air", only_air)

    # Single liquid
    single_liquid = SingleLiquidNode()
    root.add_child("single_liquid", single_liquid)

    # Single with air
    single_with_air = SingleWithAirNode()
    single_with_air.add_child("gel_dominant", GelDominantNode())
    single_with_air.add_child("stable_dominant", StableDominantNode())
    single_with_air.add_child("mixed_liquid", MixedLiquidNode())
    single_liquid.add_child("single_with_air", single_with_air)

    # Single no air
    single_no_air = SingleNoAirNode()
    single_no_air.add_child("no_air_with_lines", NoAirWithLinesNode())
    single_no_air.add_child("no_air_with_curve", NoAirWithCurveNode())
    single_no_air.add_child("no_air_with_single_line", NoAirWithSingleLineNode())
    single_no_air.add_child("no_indicators", NoAirNoIndicatorsNode())
    single_liquid.add_child("single_no_air", single_no_air)

    # Multiple liquids
    multi = MultipleLiquidsNode()
    multi_with_air = MultipleWithAirNode()
    multi_no_air = MultipleNoAirNode()
    multi.add_child("multiple_with_air", multi_with_air)
    multi.add_child("multiple_no_air", multi_no_air)
    # allow re-route to existing single_liquid logic after reclassification
    multi_no_air.add_child("route_single", single_liquid)
    root.add_child("multiple_liquids", multi)

    return root


# ============================================================================
# CLASSIFIER
# ============================================================================
class VialStateClassifierV2:
    """V2 classifier using decision tree."""

    def __init__(self):
        """Initialize classifier with decision tree."""
        self.root = _build_tree()

    def classify(self,
                crop_path: Path,
                label_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Classify vial state from crop and optional labels.

        Args:
            crop_path: Path to cropped vial image
            label_path: Optional path to YOLO label file

        Returns:
            Dictionary with classification results
        """
        # Collect evidence
        evidence = self._collect_evidence(crop_path, label_path)

        # Traverse tree
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


    def _recompute_evidence_counts(self, ev: Evidence) -> None:
        ev.liquid_count = sum(1 for d in ev.detections if d['class_id'] in LIQUID_CLASSES)
        ev.gel_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.stable_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['STABLE'])
        ev.air_count = sum(1 for d in ev.detections if d['class_id'] == CLASS_IDS['AIR'])

        ev.total_liquid_area = sum(d['area'] for d in ev.detections if d['class_id'] in LIQUID_CLASSES)
        gel_area = sum(d['area'] for d in ev.detections if d['class_id'] == CLASS_IDS['GEL'])
        ev.gel_area_fraction = (gel_area / ev.total_liquid_area) if ev.total_liquid_area > 0 else 0.0


    def _collect_evidence(self,
                         crop_path: Path,
                         label_path: Optional[Path] = None) -> Evidence:
        """Collect all evidence from detections and analyzers."""
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop image not found: {crop_path}")

        # Load image for geometry
        img = cv2.imread(str(crop_path))
        if img is None:
            raise ValueError(f"Failed to load image: {crop_path}")

        height, width = img.shape[:2]

        # Parse detections
        detections = parse_detections(label_path, width, height) if label_path and label_path.exists() else []

        # Filter low confidence
        detections = [d for d in detections if d['confidence'] > DETECTION_FILTERS["conf_min"]]

        # normalize detections and attach fractional geometry
        norm_detections = []
        img_area = float(width * height)

        for det in detections:
            # guarantee pxl xyxy
            xyxy = ensure_xyxy_px(det, width, height)
            x1, y1, x2, y2 = xyxy

            # pixel geometry
            w_px = max(0.0, x2 - x1)
            h_px = max(0.0, y2 - y1)
            area_px = w_px * h_px

            # fractional geometry in [0..1]
            box_frac = [x1 / width, y1 / height, x2 / width, y2 / height]
            area_frac = (area_px / img_area) if img_area > 0 else 0.0
            cx_frac = ((x1 + x2) * 0.5) / width
            cy_frac = ((y1 + y2) * 0.5) / height

            # uniform schema
            det_norm = {
                'class_id': det['class_id'],
                'confidence': det.get('confidence', 1.0),
                'box': [x1, y1, x2, y2],  # pixel coords for drawing & rules
                'box_frac': box_frac,  # fractional coords
                'area': area_px,  # pixel area
                'area_frac': area_frac,  # fractional area
                'center_frac': [cx_frac, cy_frac]  # ordering by height
            }
            norm_detections.append(det_norm)

        detections = norm_detections

        # Recalculate after correction
        gel_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['GEL'])
        stable_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['STABLE'])
        air_count = sum(1 for d in detections if d['class_id'] == CLASS_IDS['AIR'])
        liquid_count = gel_count + stable_count

        # Define areas
        total_liquid_area = sum(d['area'] for d in detections if d['class_id'] in LIQUID_CLASSES)
        gel_area = sum(d['area'] for d in detections if d['class_id'] == CLASS_IDS['GEL'])
        gel_area_fraction = gel_area / total_liquid_area if total_liquid_area > 0 else 0.0

        # Geometry
        cap_level_y = int(height * LINE_RULES["cap_level_frac"])  # from top
        interior_width = width

        # Line detection
        detector = LineDetector(
            min_line_length=LINE_PARAMS["min_line_length"],
            merge_threshold=LINE_PARAMS["merge_threshold"],
            horiz_kernel_div=LINE_PARAMS["horiz_kernel_div"],
            vert_kernel_div=LINE_PARAMS["vert_kernel_div"],
            adaptive_block=LINE_PARAMS["adaptive_block"],
            adaptive_c=LINE_PARAMS["adaptive_c"],
            min_line_strength=LINE_PARAMS["min_line_strength"]
        )
        line_result = detector.detect(image_path=crop_path,
                                      top_exclusion=LINE_PARAMS["top_exclusion"],
                                      bottom_exclusion=LINE_PARAMS["bottom_exclusion"]
                                      )

        # Extract horizontal line data
        horizontal_lines = (line_result.get("horizontal_lines") or {}).get("lines") or []
        vertical_lines = (line_result.get("vertical_lines") or {}).get("lines") or []
        num_horizontal_lines = len(horizontal_lines)
        num_vertical_lines = len(vertical_lines)
        # Normalize y position
        y_is_normalized = line_result.get("horizontal_lines", {}).get("y_normalized", False)
        hline_y = [(l["y_position"] * height if y_is_normalized else l["y_position"]) for l in horizontal_lines]
        # print("\nHLINEY\n", hline_y)
        hline_len_frac = [l["x_length_frac"] for l in horizontal_lines]
        hline_thickness = [l["thickness"] for l in horizontal_lines]
        hline_x_start = [l["x_start"] for l in horizontal_lines]
        hline_x_end = [l["x_end"] for l in horizontal_lines]

        # Curve analysis
        curve_result = run_curve_metrics(crop_path) or {}
        curve_stats = curve_result.get("stats", {}) or {}
        # print("\nCURVE RESULT\n", curve_result)
        # print("\nCURVE STATS\n", curve_stats)
        curve_variance = curve_stats.get("variance", curve_stats.get("variance_from_baseline", 0.0))
        curve_std_dev = curve_stats.get("std_dev", curve_stats.get("std_dev_from_baseline", 0.0))
        curve_roughness = curve_stats.get("roughness", 0.0)
        # print("\nVariance\n", curve_variance)
        # print("\nSTD DEV\n", curve_std_dev)
        # print("\nROUGH\n", curve_roughness)

        # Create evidence
        evidence = Evidence(
            detections=detections,
            liquid_count=liquid_count,
            gel_count=gel_count,
            stable_count=stable_count,
            air_count=air_count,
            total_liquid_area=total_liquid_area,
            gel_area_fraction=gel_area_fraction,
            image_height=height,
            image_width=width,
            cap_level_y=cap_level_y,
            interior_width=interior_width,
            horizontal_lines=line_result["horizontal_lines"],
            vertical_lines=line_result["vertical_lines"],
            num_horizontal_lines=num_horizontal_lines,
            num_vertical_lines=num_vertical_lines,
            hline_y=hline_y,
            hline_len_frac=hline_len_frac,
            hline_line_thickness=hline_thickness,
            hline_x_start=hline_x_start,
            hline_x_end=hline_x_end,
            curve_stats=curve_stats,
            curve_variance=curve_variance,
            curve_std_dev=curve_std_dev,
            curve_roughness=curve_roughness,
            image_path=crop_path
        )

        # print("\nEVIDENCE\n", evidence)
        # print("\nAIR BEFORE\n", evidence.air_count)
        # print("\nLIQUIDS BEFORE\n", evidence.liquid_count)
        # print("\nGEL BEFORE\n", evidence.gel_count)

        _apply_air_placement_rules(evidence)
        self._recompute_evidence_counts(evidence)

        # print("\nAIR AFTER\n", evidence.air_count)
        # print("\nLIQUIDS AFTER\n", evidence.liquid_count)
        # print("\nGEL AFTER\n", evidence.gel_count)

        return evidence

    def _traverse_tree(self, evidence: Evidence) -> Decision:
        """Traverse the tree from root."""
        current_node = self.root
        path = []
        depth = 0
        max_depth = 20  # prevent infinite loops

        while depth < max_depth:
            next_key, decision = current_node.evaluate(evidence, path)

            if decision is not None:
                # Terminal node
                return decision

            if next_key not in current_node.children:
                # No valid path, return unknown
                return Decision(
                    state=VialState.UNKNOWN,
                    confidence=Confidence.LOW,
                    reasoning=[f"Tree traversal failed at {current_node.name}"],
                    node_path=path
                )

            current_node = current_node.children[next_key]
            depth += 1

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

    highlight_path: list of node names visited in order.
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

    if highlight_path and len(highlight_path) >= 1:
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

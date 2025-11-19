import sys
from dataclasses import dataclass
from pathlib import Path

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from typing import Tuple, Dict, Any, List

from config import TURBIDITY_PARAMS

@dataclass
class SeparationEvent:
    type: str                 # separation interface type
    boundary_norm: float      # normalized y position of the boundary
    boundary_pixel: int       # pixel position of the boundary
    top_phase: str            # phase of upper segment
    bottom_phase: str         # phase of lower segment
    delta_brightness: float   # absolute difference in mean brightness
    top_segment: Dict[str, Any] # upper segment data
    bottom_segment: Dict[str, Any] # lower segment data

    # convert to dict for json serialization
    def to_dict(self):
        return {
            "type": self.type,
            "boundary_norm": float(self.boundary_norm),
            "boundary_pixel": int(self.boundary_pixel),
            "top_phase": self.top_phase,
            "bottom_phase": self.bottom_phase,
            "delta_brightness": float(self.delta_brightness),
            "top_segment": self.top_segment,
            "bottom_segment": self.bottom_segment,
        }


def detect_phase_separation_from_separations(
        separation_events: List[SeparationEvent],
        min_liquid_interfaces: int = TURBIDITY_PARAMS["min_liquid_interfaces"],
        min_vertical_span: float = TURBIDITY_PARAMS["min_vertical_span"]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide if the vial is phase-separated based on separation events
    """
    if not separation_events:
        return False, {"reason": "no_separation_events"}

    liquid_types = {"opaque-translucent", "translucent-opaque", "liquid-liquid"}

    # collect all liquid-liquid-type interfaces
    liquid_ifaces = [event for event in separation_events if event.type in liquid_types]
    air_iface = [event for event in separation_events if event.type == "air-liquid"]

    if len(liquid_ifaces) < min_liquid_interfaces:
        return False, {
            "reason": "too_few_liquid_interfaces",
            "num_liquid_interfaces": len(liquid_ifaces),
            "interfaces": liquid_ifaces,
            "air_interface": air_iface,
        }

    # check vertical span
    ys = [ev.boundary_norm for ev in liquid_ifaces]
    span = max(ys) - min(ys) if ys else 0.0

    # if span < min_vertical_span:
    #     return False, {
    #         "reason": "liquid_interfaces_span_too_small",
    #         "num_liquid_interfaces": len(liquid_ifaces),
    #         "span": span,
    #     }

    return True, {
        "num_liquid_interfaces": len(liquid_ifaces),
        "span": span,
        "interfaces": liquid_ifaces,
        "air_interface": air_iface,
    }


def classify_segment_phase(mean_brightness: float) -> str:
    """
    Classify a segment into AIR / LIQUID_TRANSLUCENT / LIQUID_OPAQUE
    based purely on mean brightness (0–1).
    Adjust thresholds in PHASE_THRESHOLDS to your dataset.
    """
    if mean_brightness <= TURBIDITY_PARAMS["air_max"]:
        return "AIR"
    elif mean_brightness <= TURBIDITY_PARAMS["translucent_max"]:
        return "LIQUID_TRANSLUCENT"
    else:
        return "LIQUID_OPAQUE"


def label_segments(
        segments: List[Dict[str, Any]],
        analysis_height: int
) -> List[Dict[str, Any]]:
    """
    Add phase labels and height fraction to each segment, enforcing:
      - air can only appear above the first liquid layer
      - any segment below a liquid that would otherwise be AIR
        is treated as LIQUID_TRANSLUCENT
    """
    labeled: List[Dict[str, Any]] = []
    seen_liquid = False  # passed first liquid layer

    for seg in segments:  # segments must be sorted top to bottom
        h_frac = seg["height_pixels"] / analysis_height
        if h_frac < TURBIDITY_PARAMS["min_segment_height_frac"]:
            # ignore thin segments
            continue

        phase = classify_segment_phase(seg["mean_brightness"])

        # no air below liquid
        if seen_liquid and phase == "AIR":
            phase = "LIQUID_TRANSLUCENT"

        if phase.startswith("LIQUID"):
            seen_liquid = True

        seg_labeled = {
            **seg,
            "phase": phase,
            "height_frac": h_frac,
        }
        labeled.append(seg_labeled)

    return labeled


def detect_separation_types(
        segments: List[Dict[str, Any]]
) -> List[SeparationEvent]:
    """
    given labeled segments, return a list of separation events
    with types:
      - 'air-liquid'
      - 'liquid-liquid'
      - 'opaque-translucent'
      - 'translucent-opaque'
    """
    events: List[SeparationEvent] = []
    if len(segments) < 2:
        return events

    contrast_min = TURBIDITY_PARAMS["liquid_liquid_contrast"]

    for upper, lower in zip(segments, segments[1:]):
        p_top = upper["phase"]
        p_bottom = lower["phase"]
        mu_top = float(upper["mean_brightness"])
        mu_bottom = float(lower["mean_brightness"])
        delta_mu = abs(mu_bottom - mu_top)

        # use end of upper segment as boundary position
        boundary_norm = float(upper.get("end_normalized", upper.get("start_normalized", 0.0)))
        boundary_pixel = int(boundary_norm * TURBIDITY_PARAMS["analysis_height"])
        # get boundaries for full image height
        boundary_norm_full_img = boundary_norm * TURBIDITY_PARAMS["analysis_height"] / TURBIDITY_PARAMS["analysis_height"]

        # skip small contrast differences
        if delta_mu < contrast_min:
            continue

        # determine separation type
        if "AIR" in (p_top, p_bottom) and (p_top != p_bottom):
            sep_type = "air-liquid"

        elif {p_top, p_bottom} == {"LIQUID_OPAQUE", "LIQUID_TRANSLUCENT"}:
            if p_top == "LIQUID_OPAQUE" and p_bottom == "LIQUID_TRANSLUCENT":
                sep_type = "opaque-translucent"
            else:
                sep_type = "translucent-opaque"

        # elif p_top.startswith("LIQUID") and p_bottom.startswith("LIQUID"):
        #     # both liquids, brightness different enough
        #     sep_type = "liquid-liquid"
        else:
            # same phase or unsupported combination
            continue

        # use end of upper segment as boundary position
        boundary_norm = float(upper.get("end_normalized", upper.get("start_normalized", 0.0)))

        ev = SeparationEvent(
            type=sep_type,
            boundary_norm=boundary_norm,
            boundary_pixel=boundary_pixel,
            top_phase=p_top,
            bottom_phase=p_bottom,
            delta_brightness=delta_mu,
            top_segment=upper,
            bottom_segment=lower,
        )
        events.append(ev)

    return events

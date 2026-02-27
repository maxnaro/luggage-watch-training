"""
Test inference script with abandonment detection logic.

Implements the three algorithms from the report:
  1. Luggage ownership association (nearest-person within radius)
  2. Luggage state transitions  (attended -> unattended -> abandoned)
  3. Alert generation           (log + on-frame overlay)

Usage:
  python test_inference.py --weights <path> --source <video> [options]
"""

import argparse
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = (
    "C:\\source\\luggage-watch\\training\\runs\\detect\\runs"
    "\\yolo11m_100226\\weights\\best.pt"
)
DEFAULT_SOURCE = "C:\\source\\AbandonObjVideo\\AVSS_E2.avi"
DEFAULT_TRACKER = "botsort.yaml"
DEFAULT_OWNERSHIP_RADIUS = 150  # pixels
DEFAULT_ABANDON_THRESHOLD = 30.0  # seconds
DEFAULT_STATIONARY_THRESHOLD = 10  # pixels – max centre drift to count as stationary


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------
class LuggageState(Enum):
    ATTENDED = auto()
    UNATTENDED = auto()
    ABANDONED = auto()


@dataclass
class TrackedLuggage:
    """Mutable record for a single piece of tracked luggage."""

    track_id: int
    state: LuggageState = LuggageState.ATTENDED
    owner_id: int | None = None
    timer_start: float | None = None
    last_centre: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)  # x1 y1 x2 y2


@dataclass
class Alert:
    """Immutable record emitted when luggage becomes abandoned."""

    timestamp: float
    luggage_id: int
    location: tuple[float, float]
    duration: float
    last_owner: int | None
    bbox: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (cx, cy) from an (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Algorithm 1 – ownership association
# ---------------------------------------------------------------------------
def associate_owner(
    luggage_centre: tuple[float, float],
    persons: dict[int, tuple[float, float, float, float]],
    radius: float,
) -> int | None:
    """Return the track-ID of the nearest person within *radius*, or None."""
    min_dist = float("inf")
    best: int | None = None
    for pid, pbbox in persons.items():
        d = euclidean(luggage_centre, centre(pbbox))
        if d < min_dist and d < radius:
            min_dist = d
            best = pid
    return best


def is_owner_nearby(
    owner_id: int | None,
    luggage_centre: tuple[float, float],
    persons: dict[int, tuple[float, float, float, float]],
    radius: float,
) -> bool:
    """Return True if the specific *owner_id* is within *radius* of the luggage."""
    if owner_id is None or owner_id not in persons:
        return False
    return euclidean(luggage_centre, centre(persons[owner_id])) < radius


# ---------------------------------------------------------------------------
# Algorithm 3 – alert generation
# ---------------------------------------------------------------------------
def generate_alert(lug: TrackedLuggage, now: float) -> Alert:
    """Build an Alert for an abandoned luggage item and log it."""
    loc = centre(lug.bbox)
    duration = now - (lug.timer_start or now)
    alert = Alert(
        timestamp=now,
        luggage_id=lug.track_id,
        location=loc,
        duration=duration,
        last_owner=lug.owner_id,
        bbox=lug.bbox,
    )
    print(
        f"[ALERT] Luggage {alert.luggage_id} ABANDONED  "
        f"duration={alert.duration:.1f}s  "
        f"location=({alert.location[0]:.0f}, {alert.location[1]:.0f})  "
        f"last_owner={alert.last_owner}"
    )
    return alert


# ---------------------------------------------------------------------------
# Algorithm 2 – state transitions  (called once per frame)
# ---------------------------------------------------------------------------
def update_states(
    luggage_detections: dict[int, tuple[float, float, float, float]],
    person_detections: dict[int, tuple[float, float, float, float]],
    state_store: dict[int, TrackedLuggage],
    alerts: list[Alert],
    radius: float,
    abandon_threshold: float,
    stationary_threshold: float,
    now: float,
) -> None:
    """Apply the state-machine logic to every tracked luggage item."""

    # Mark luggage that has disappeared from the frame for cleanup
    active_ids = set(luggage_detections.keys())
    lost_ids = [tid for tid in state_store if tid not in active_ids]
    for tid in lost_ids:
        del state_store[tid]

    for lid, bbox in luggage_detections.items():
        lc = centre(bbox)
        owner = associate_owner(lc, person_detections, radius)

        # Initialise new tracks
        if lid not in state_store:
            state_store[lid] = TrackedLuggage(
                track_id=lid,
                state=LuggageState.ATTENDED,
                owner_id=owner,
                last_centre=lc,
                bbox=bbox,
            )
            continue

        lug = state_store[lid]
        is_stationary = euclidean(lc, lug.last_centre) < stationary_threshold
        lug.last_centre = lc
        lug.bbox = bbox

        # Check whether the *original* owner is still / again nearby
        owner_nearby = is_owner_nearby(lug.owner_id, lc, person_detections, radius)

        match lug.state:
            case LuggageState.ATTENDED:
                # First frame: lock in the nearest person as owner
                if lug.owner_id is None and owner is not None:
                    lug.owner_id = owner

                if not owner_nearby and is_stationary:
                    lug.state = LuggageState.UNATTENDED
                    lug.timer_start = now
                # Owner is nearby – remain attended (owner_id unchanged)

            case LuggageState.UNATTENDED:
                if owner_nearby or not is_stationary:
                    # Original owner returned or luggage moved
                    lug.state = LuggageState.ATTENDED
                    lug.timer_start = None
                elif (
                    lug.timer_start is not None
                    and now - lug.timer_start >= abandon_threshold
                ):
                    lug.state = LuggageState.ABANDONED
                    alerts.append(generate_alert(lug, now))

            case LuggageState.ABANDONED:
                # Remain abandoned – operator must resolve
                pass


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
STATE_COLOURS = {
    LuggageState.ATTENDED: (0, 200, 0),  # green
    LuggageState.UNATTENDED: (0, 200, 255),  # orange
    LuggageState.ABANDONED: (0, 0, 255),  # red
}


def draw_overlays(
    frame,
    state_store: dict[int, TrackedLuggage],
) -> None:
    """Draw bounding boxes and state labels onto the frame."""
    for lug in state_store.values():
        x1, y1, x2, y2 = map(int, lug.bbox)
        colour = STATE_COLOURS[lug.state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = f"ID:{lug.track_id} {lug.state.name}"
        if lug.state == LuggageState.UNATTENDED and lug.timer_start is not None:
            elapsed = time.time() - lug.timer_start
            label += f" {elapsed:.0f}s"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Resolve class-name → class-id mapping from the model
# ---------------------------------------------------------------------------
def resolve_class_ids(model: YOLO) -> tuple[set[int], set[int]]:
    """
    Return (person_class_ids, luggage_class_ids) by inspecting model.names.
    Handles both COCO-pretrained and custom fine-tuned label sets.
    """
    person_keywords = {"person", "people", "pedestrian"}
    luggage_keywords = {
        "luggage",
        "bag",
        "suitcase",
        "backpack",
        "handbag",
        "briefcase",
    }

    person_ids: set[int] = set()
    luggage_ids: set[int] = set()

    for cid, name in model.names.items():
        lower = name.lower().strip()
        if lower in person_keywords:
            person_ids.add(cid)
        elif lower in luggage_keywords:
            luggage_ids.add(cid)

    if not person_ids or not luggage_ids:
        print(f"[WARN] Model class names: {model.names}")
        print(f"       Resolved person={person_ids}, luggage={luggage_ids}")
        print("       Check that your model has the expected classes.")

    return person_ids, luggage_ids


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Test inference with abandonment logic")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pt weights")
    ap.add_argument(
        "--source", default=DEFAULT_SOURCE, help="Video file or camera index"
    )
    ap.add_argument("--tracker", default=DEFAULT_TRACKER, help="Tracker config YAML")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_OWNERSHIP_RADIUS,
        help="Max pixel distance for ownership association",
    )
    ap.add_argument(
        "--abandon-time",
        type=float,
        default=DEFAULT_ABANDON_THRESHOLD,
        help="Seconds before unattended luggage becomes abandoned",
    )
    ap.add_argument(
        "--stationary-thresh",
        type=float,
        default=DEFAULT_STATIONARY_THRESHOLD,
        help="Max centre drift (px) to count as stationary",
    )
    ap.add_argument("--no-show", action="store_true", help="Disable cv2 display window")
    args = ap.parse_args()

    model = YOLO(args.weights)
    person_ids, luggage_ids = resolve_class_ids(model)
    print(f"[INFO] Person classes: {person_ids}  Luggage classes: {luggage_ids}")

    state_store: dict[int, TrackedLuggage] = {}
    alerts: list[Alert] = []

    results = model.track(
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        stream=True,
        show=False,  # we handle display ourselves
        conf=args.conf,
        persist=True,
        tracker=args.tracker,
    )

    for r in results:
        frame = r.orig_img.copy()
        now = time.time()

        # Partition detections by class
        persons: dict[int, tuple[float, float, float, float]] = {}
        luggage: dict[int, tuple[float, float, float, float]] = {}

        if r.boxes is not None and r.boxes.id is not None:
            for box, cls_id, track_id in zip(
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.cls.cpu().tolist(),
                r.boxes.id.cpu().tolist(),
            ):
                tid = int(track_id)
                cid = int(cls_id)
                bbox_tuple = tuple(box)
                if cid in person_ids:
                    persons[tid] = bbox_tuple
                elif cid in luggage_ids:
                    luggage[tid] = bbox_tuple

        # Algorithm 2: update state machine
        update_states(
            luggage_detections=luggage,
            person_detections=persons,
            state_store=state_store,
            alerts=alerts,
            radius=args.radius,
            abandon_threshold=args.abandon_time,
            stationary_threshold=args.stationary_thresh,
            now=now,
        )

        # Draw overlays and display
        draw_overlays(frame, state_store)

        if not args.no_show:
            cv2.imshow("Luggage Watch", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"\n[INFO] Session complete. Total alerts generated: {len(alerts)}")
    for a in alerts:
        print(
            f"  Luggage {a.luggage_id}: abandoned at "
            f"({a.location[0]:.0f}, {a.location[1]:.0f}), "
            f"duration={a.duration:.1f}s, last_owner={a.last_owner}"
        )


if __name__ == "__main__":
    main()

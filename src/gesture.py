from __future__ import annotations

import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LARGE_DISTANCE = 9999.0
EXPECTED_LANDMARK_COUNT = 21
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "hand_landmarker.task"


def _as_landmark_dict(landmark: Any) -> dict[str, float]:
    """Accept dict-like or MediaPipe landmark objects and return x/y/z floats."""
    if isinstance(landmark, dict):
        return {
            "x": float(landmark.get("x", 0.0)),
            "y": float(landmark.get("y", 0.0)),
            "z": float(landmark.get("z", 0.0)),
        }

    return {
        "x": float(getattr(landmark, "x", 0.0)),
        "y": float(getattr(landmark, "y", 0.0)),
        "z": float(getattr(landmark, "z", 0.0)),
    }


def _extract_landmark_list(hand_landmarks: Any) -> list[dict[str, float]]:
    if isinstance(hand_landmarks, dict) and "landmarks" in hand_landmarks:
        hand_landmarks = hand_landmarks["landmarks"]
    if hasattr(hand_landmarks, "landmark"):
        hand_landmarks = hand_landmarks.landmark
    if not hand_landmarks:
        return []
    return [_as_landmark_dict(landmark) for landmark in hand_landmarks]


def normalize_hand_landmarks(hand_landmarks: Any) -> dict[str, Any]:
    """Normalize one hand using wrist as anchor and hand size as scale.

    x/y are made relative to wrist (landmark 0). z is also scaled, but matching
    focuses on x/y because z is often noisy on ordinary webcams.
    """
    landmarks = _extract_landmark_list(hand_landmarks)
    if len(landmarks) < EXPECTED_LANDMARK_COUNT:
        raise ValueError(f"Expected {EXPECTED_LANDMARK_COUNT} hand landmarks, got {len(landmarks)}")

    points = np.array([[point["x"], point["y"], point["z"]] for point in landmarks], dtype=np.float32)
    wrist = points[0]

    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    bbox_scale = float(max(max_xy - min_xy))
    wrist_to_middle_mcp = float(np.linalg.norm(points[9, :2] - wrist[:2]))
    scale = max(bbox_scale, wrist_to_middle_mcp, 1e-6)

    normalized_points = (points - wrist) / scale
    center = points[:, :2].mean(axis=0)

    return {
        "landmarks": [
            {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            for point in normalized_points
        ],
        "center": {"x": float(center[0]), "y": float(center[1])},
        "wrist": {"x": float(wrist[0]), "y": float(wrist[1])},
        "scale": float(scale),
    }


def normalize_hands_data(hands_data: list[dict]) -> dict[str, Any]:
    """Normalize all detected hands and include relative two-hand position."""
    if not hands_data:
        return {"version": 1, "hand_count": 0, "hands": [], "relative_position": None}

    normalized_hands: list[dict[str, Any]] = []
    for hand in hands_data[:2]:
        normalized = normalize_hand_landmarks(hand.get("landmarks", hand))
        normalized["handedness"] = hand.get("handedness") if isinstance(hand, dict) else None
        normalized["score"] = hand.get("score") if isinstance(hand, dict) else None
        normalized_hands.append(normalized)

    relative_position = None
    if len(normalized_hands) == 2:
        relative_position = _relative_position(normalized_hands[0], normalized_hands[1])

    return {
        "version": 1,
        "hand_count": len(normalized_hands),
        "hands": normalized_hands,
        "relative_position": relative_position,
    }


def save_scuba_pose_template(hands_data: list[dict], path: str, source: str = "webcam") -> dict[str, Any]:
    """Save current detected hands as SCUBA_POSE template JSON."""
    if not hands_data:
        raise ValueError("Cannot save template: no hands detected.")

    template_data = normalize_hands_data(hands_data)
    template_data["gesture"] = "SCUBA_POSE"
    template_data["source"] = source
    template_data["created_at"] = datetime.now().isoformat(timespec="seconds")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(template_data, file, indent=2)

    return template_data


def load_scuba_pose_template(path: str) -> dict[str, Any] | None:
    """Load template if available. Return None instead of crashing on error."""
    template_path = Path(path)
    if not template_path.exists():
        return None

    try:
        with template_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read template {template_path}: {exc}")
        return None

    if not isinstance(data, dict) or not data.get("hands"):
        print(f"Invalid template file: {template_path}")
        return None

    return data


def _ensure_normalized(data: Any) -> dict[str, Any] | None:
    if data is None:
        return None
    if isinstance(data, dict) and "hands" in data:
        return data
    if isinstance(data, list):
        try:
            return normalize_hands_data(data)
        except ValueError:
            return None
    return None


def _landmark_distance(hand_a: dict[str, Any], hand_b: dict[str, Any]) -> float:
    landmarks_a = hand_a.get("landmarks", [])
    landmarks_b = hand_b.get("landmarks", [])
    if len(landmarks_a) != EXPECTED_LANDMARK_COUNT or len(landmarks_b) != EXPECTED_LANDMARK_COUNT:
        return LARGE_DISTANCE

    distances = []
    for point_a, point_b in zip(landmarks_a, landmarks_b):
        dx = float(point_a["x"]) - float(point_b["x"])
        dy = float(point_a["y"]) - float(point_b["y"])
        distances.append(math.sqrt(dx * dx + dy * dy))
    return float(np.mean(distances))


def _relative_position(hand_a: dict[str, Any], hand_b: dict[str, Any]) -> dict[str, float]:
    center_a = hand_a["center"]
    center_b = hand_b["center"]
    average_scale = max((float(hand_a.get("scale", 1.0)) + float(hand_b.get("scale", 1.0))) / 2.0, 1e-6)
    return {
        "dx": (float(center_b["x"]) - float(center_a["x"])) / average_scale,
        "dy": (float(center_b["y"]) - float(center_a["y"])) / average_scale,
    }


def _relative_position_distance(current_hands_in_template_order: list[dict], template_hands: list[dict]) -> float:
    if len(current_hands_in_template_order) != 2 or len(template_hands) != 2:
        return 0.0

    current_relative = _relative_position(current_hands_in_template_order[0], current_hands_in_template_order[1])
    template_relative = _relative_position(template_hands[0], template_hands[1])
    dx = current_relative["dx"] - template_relative["dx"]
    dy = current_relative["dy"] - template_relative["dy"]
    return math.sqrt(dx * dx + dy * dy)


def _mirror_normalized_pose(pose_data: dict[str, Any]) -> dict[str, Any]:
    """Return a mirrored copy of normalized pose data for selfie/photo differences."""
    mirrored_hands: list[dict[str, Any]] = []
    for hand in pose_data.get("hands", []):
        mirrored_hand = dict(hand)
        mirrored_hand["landmarks"] = [
            {"x": -float(point["x"]), "y": float(point["y"]), "z": float(point.get("z", 0.0))}
            for point in hand.get("landmarks", [])
        ]
        if "center" in hand:
            mirrored_hand["center"] = {"x": -float(hand["center"]["x"]), "y": float(hand["center"]["y"])}
        if "wrist" in hand:
            mirrored_hand["wrist"] = {"x": -float(hand["wrist"]["x"]), "y": float(hand["wrist"]["y"])}
        mirrored_hands.append(mirrored_hand)

    mirrored_pose = dict(pose_data)
    mirrored_pose["hands"] = mirrored_hands
    return mirrored_pose


def _calculate_pose_distance_normalized(current: dict[str, Any], template: dict[str, Any]) -> float:
    current_hands = current.get("hands", [])
    template_hands = template.get("hands", [])
    if not current_hands or not template_hands:
        return LARGE_DISTANCE
    if len(current_hands) != len(template_hands):
        return LARGE_DISTANCE

    hand_count = len(template_hands)
    best_distance = LARGE_DISTANCE
    for current_order in itertools.permutations(range(hand_count)):
        ordered_current_hands = [current_hands[index] for index in current_order]
        hand_distances = [
            _landmark_distance(ordered_current_hands[index], template_hands[index])
            for index in range(hand_count)
        ]
        landmark_score = float(np.mean(hand_distances))

        # Relative two-hand location helps, but keep the weight small because
        # a reference photo and a live webcam often have different perspective.
        relative_score = _relative_position_distance(ordered_current_hands, template_hands)
        total_score = landmark_score + (0.03 * relative_score)
        best_distance = min(best_distance, total_score)

    return float(best_distance)


def calculate_pose_distance(current_hands_data: Any, template_data: Any) -> float:
    """Calculate average normalized Euclidean distance to a template.

    For two-hand templates, both hand orders are tested so swapped Left/Right
    order does not break matching. Smaller distance means more similar.
    """
    current = _ensure_normalized(current_hands_data)
    template = _ensure_normalized(template_data)
    if not current or not template:
        return LARGE_DISTANCE

    return min(
        _calculate_pose_distance_normalized(current, template),
        _calculate_pose_distance_normalized(_mirror_normalized_pose(current), template),
    )


def is_scuba_pose(current_hands_data: Any, template_data: Any, threshold: float) -> bool:
    if not current_hands_data or not template_data:
        return False
    return calculate_pose_distance(current_hands_data, template_data) <= threshold


def extract_hands_from_image(image_path: str) -> list[dict]:
    """Run MediaPipe Hands on a reference image and return clean hand data."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError("assets/models/hand_landmarker.task")

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(str(path))

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = np.ascontiguousarray(rgb_image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return []

    hands_data: list[dict] = []
    for index, hand_landmarks in enumerate(results.hand_landmarks):
        handedness_label = None
        handedness_score = None
        if results.handedness and index < len(results.handedness) and results.handedness[index]:
            classification = results.handedness[index][0]
            handedness_label = classification.category_name
            handedness_score = float(classification.score)

        hands_data.append(
            {
                "landmarks": [_as_landmark_dict(landmark) for landmark in hand_landmarks],
                "handedness": handedness_label,
                "score": handedness_score,
            }
        )

    return hands_data


def save_scuba_pose_template_from_image(image_path: str, output_json_path: str) -> dict[str, Any]:
    """Create and save a SCUBA_POSE template from a reference photo."""
    hands_data = extract_hands_from_image(image_path)
    if not hands_data:
        raise ValueError("No hands detected in reference image. Try another photo with clearer hands.")
    if len(hands_data) < 2:
        print("Warning: Only one hand detected in reference image. Matching will require one hand.")

    return save_scuba_pose_template(hands_data, output_json_path, source="reference_image")
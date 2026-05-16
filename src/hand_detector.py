from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [(connection.start, connection.end) for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "models" / "hand_landmarker.task"


class HandDetector:
    """Wrapper kecil untuk MediaPipe Hands agar main.py tetap bersih."""

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        self.max_hands = max_hands
        self.timestamp_ms = 0

        if not DEFAULT_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Missing MediaPipe model: assets/models/hand_landmarker.task. "
                "Download it from the MediaPipe Hand Landmarker model page."
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame) -> list[dict]:
        """Detect hands and return clean landmark data.

        Returned hand item shape:
        {
            "landmarks": [{"x": float, "y": float, "z": float}, ...],
            "handedness": "Left" | "Right" | None,
            "score": float | None,
        }
        """
        if frame is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 33
        results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        hands_data: list[dict] = []
        if not results.hand_landmarks:
            return hands_data

        for index, hand_landmarks in enumerate(results.hand_landmarks):
            handedness_label = None
            handedness_score = None

            if results.handedness and index < len(results.handedness) and results.handedness[index]:
                classification = results.handedness[index][0]
                handedness_label = classification.category_name
                handedness_score = float(classification.score)

            landmarks = [
                {"x": float(landmark.x), "y": float(landmark.y), "z": float(landmark.z)}
                for landmark in hand_landmarks
            ]

            hands_data.append(
                {
                    "landmarks": landmarks,
                    "handedness": handedness_label,
                    "score": handedness_score,
                }
            )

        return hands_data

    def draw_landmarks(self, frame, hands_data: list[dict]) -> None:
        """Draw red landmark dots, white hand connections, and Left/Right labels."""
        if frame is None or not hands_data:
            return

        height, width = frame.shape[:2]

        for hand in hands_data:
            landmarks = hand.get("landmarks", [])
            if not landmarks:
                continue

            pixel_points = [
                (int(point["x"] * width), int(point["y"] * height))
                for point in landmarks
            ]

            for start, end in HAND_CONNECTIONS:
                if start < len(pixel_points) and end < len(pixel_points):
                    cv2.line(frame, pixel_points[start], pixel_points[end], (255, 255, 255), 2, cv2.LINE_AA)

            for point in pixel_points:
                cv2.circle(frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA)

            min_x = int(min(point["x"] for point in landmarks) * width)
            min_y = int(min(point["y"] for point in landmarks) * height)
            label = hand.get("handedness") or "Hand"
            score = hand.get("score")
            if score is not None:
                label = f"{label} {score:.2f}"

            text_position = (max(min_x, 5), max(min_y - 10, 20))
            cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        self.landmarker.close()
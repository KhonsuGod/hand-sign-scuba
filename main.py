from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.gesture import (
    LARGE_DISTANCE,
    calculate_pose_distance,
    load_scuba_pose_template,
    save_scuba_pose_template,
    save_scuba_pose_template_from_image,
)
from src.hand_detector import HandDetector
from src.video_popup import VideoPopupManager


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "config"
TEMPLATE_PATH = CONFIG_DIR / "scuba_pose.json"
REFERENCE_IMAGE_PATH = ROOT_DIR / "assets" / "reference" / "scuba_pose_reference.jpg"
VIDEO_PATHS = [
    ROOT_DIR / "assets" / "videos" / "scuba_1.mp4",
    ROOT_DIR / "assets" / "videos" / "scuba_2.mp4",
]

WINDOW_TITLE = "Hand Sign Scuba"
DEFAULT_THRESHOLD = 0.55
THRESHOLD_STEP = 0.05
MIN_THRESHOLD = 0.02
MAX_THRESHOLD = 2.00
POSE_STABLE_SECONDS = 0.50
POSE_HIDE_SECONDS = 2.00


def rel_path(path: Path) -> str:
    """Return a user-friendly path relative to this project root."""
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def template_source_label(template_data: dict | None) -> str:
    if not template_data:
        return "None"

    source = template_data.get("source", "unknown")
    if source == "webcam":
        return "Webcam"
    if source == "reference_image":
        return "Reference Image"
    return "Loaded"


def draw_overlay(
    frame,
    fps: float,
    hand_count: int,
    template_data: dict | None,
    gesture_name: str,
    distance: float | None,
    threshold: float,
    video_manager: VideoPopupManager,
    last_action_message: str,
) -> None:
    template_status = "Loaded" if template_data else "Not Found"
    source = template_source_label(template_data)
    video_status = "Showing" if video_manager.is_showing() else "Hidden"
    distance_text = "N/A" if distance is None or distance >= LARGE_DISTANCE else f"{distance:.3f}"

    left_lines = [
        f"FPS {fps:.0f} | Hands {hand_count}",
        f"Gesture: {gesture_name}",
        f"Dist {distance_text} / Thr {threshold:.2f}",
        f"Template: {template_status} ({source})",
        f"Video: {video_status}",
    ]
    if last_action_message:
        left_lines.append(f"Last: {last_action_message[:32]}")

    right_lines = [
        "C webcam calib",
        "L ref image",
        "B show 2 memes",
        "1/2 test video",
        "H hide | Q quit",
        "+/- threshold",
    ]

    def put_small_text(lines: list[str], x: int, y: int) -> None:
        line_height = 17
        for index, line in enumerate(lines):
            y_pos = y + index * line_height
            cv2.putText(frame, line, (x + 1, y_pos + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    height, width = frame.shape[:2]
    put_small_text(left_lines, 10, 18)
    put_small_text(right_lines, max(10, width - 145), 18)


def load_template_at_startup() -> tuple[dict | None, str]:
    template_data = load_scuba_pose_template(str(TEMPLATE_PATH))
    if template_data:
        print(f"Loaded template: {rel_path(TEMPLATE_PATH)}")
        return template_data, "Template loaded from config"

    if REFERENCE_IMAGE_PATH.exists():
        message = "Press L to create template from reference image or Press C to calibrate using webcam"
    else:
        message = "Template not found. Press C to calibrate or L after adding reference image."
    print(message)
    return None, message


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    threshold = DEFAULT_THRESHOLD
    template_data, last_action_message = load_template_at_startup()

    detector = HandDetector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.7)
    video_manager = VideoPopupManager([str(path) for path in VIDEO_PATHS])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam tidak terbuka. Coba tutup aplikasi lain yang memakai kamera.")
        detector.close()
        video_manager.release()
        return

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    fps = 0.0
    previous_time = time.time()
    scuba_detected_since: float | None = None
    last_scuba_seen_at: float | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Gagal membaca frame webcam.")
                break

            now = time.time()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now
            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps == 0.0 else (0.90 * fps + 0.10 * instant_fps)

            hands_data = detector.detect(frame)
            detector.draw_landmarks(frame, hands_data)

            distance = None
            gesture_name = "UNKNOWN"
            is_scuba = False

            if template_data and hands_data:
                distance = calculate_pose_distance(hands_data, template_data)
                is_scuba = distance <= threshold
                gesture_name = "SCUBA_POSE" if is_scuba else "UNKNOWN"

            if is_scuba:
                if scuba_detected_since is None:
                    scuba_detected_since = now
                last_scuba_seen_at = now
                if now - scuba_detected_since >= POSE_STABLE_SECONDS:
                    video_manager.show_all()
            else:
                scuba_detected_since = None
                if (
                    not video_manager.manual_mode
                    and last_scuba_seen_at is not None
                    and now - last_scuba_seen_at >= POSE_HIDE_SECONDS
                    and video_manager.is_showing()
                ):
                    video_manager.hide_all()

            video_manager.update()

            draw_overlay(
                frame=frame,
                fps=fps,
                hand_count=len(hands_data),
                template_data=template_data,
                gesture_name=gesture_name,
                distance=distance,
                threshold=threshold,
                video_manager=video_manager,
                last_action_message=last_action_message,
            )

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 255:
                continue
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                if not hands_data:
                    last_action_message = "No hands detected. Cannot save webcam template."
                    print(last_action_message)
                    continue
                try:
                    template_data = save_scuba_pose_template(hands_data, str(TEMPLATE_PATH), source="webcam")
                    last_action_message = "Template saved from webcam"
                    print(f"{last_action_message}: {rel_path(TEMPLATE_PATH)}")
                except Exception as exc:
                    last_action_message = f"Failed to save webcam template: {exc}"
                    print(last_action_message)
            elif key in (ord("l"), ord("L")):
                try:
                    template_data = save_scuba_pose_template_from_image(str(REFERENCE_IMAGE_PATH), str(TEMPLATE_PATH))
                    last_action_message = "Template loaded from reference image"
                    print("Loaded scuba pose template from reference image.")
                except FileNotFoundError:
                    last_action_message = f"Missing reference image: {rel_path(REFERENCE_IMAGE_PATH)}"
                    print(last_action_message)
                except ValueError as exc:
                    last_action_message = str(exc)
                    print(last_action_message)
                except Exception as exc:
                    last_action_message = f"Failed loading reference image: {exc}"
                    print(last_action_message)
            elif key in (ord("r"), ord("R")):
                template_data = load_scuba_pose_template(str(TEMPLATE_PATH))
                last_action_message = "Template reloaded from config" if template_data else "Template not found"
                print(last_action_message)
            elif key in (ord("b"), ord("B")):
                video_manager.show_all(manual=True)
                last_action_message = "Manual test: showing 2 meme videos"
                print(last_action_message)
            elif key in (ord("h"), ord("H")):
                video_manager.hide_all()
                last_action_message = "Videos hidden"
                print(last_action_message)
            elif key == ord("1"):
                video_manager.show_one(0, manual=True)
                last_action_message = "Manual test: showing video 1"
                print(last_action_message)
            elif key == ord("2"):
                video_manager.show_one(1, manual=True)
                last_action_message = "Manual test: showing video 2"
                print(last_action_message)
            elif key in (ord("+"), ord("=")):
                threshold = min(MAX_THRESHOLD, threshold + THRESHOLD_STEP)
                last_action_message = f"Threshold increased to {threshold:.2f}"
                print(last_action_message)
            elif key in (ord("-"), ord("_")):
                threshold = max(MIN_THRESHOLD, threshold - THRESHOLD_STEP)
                last_action_message = f"Threshold decreased to {threshold:.2f}"
                print(last_action_message)

    finally:
        cap.release()
        detector.close()
        video_manager.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
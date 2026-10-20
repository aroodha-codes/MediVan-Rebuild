"""
ai_obstacle_detector.py

AI-based obstacle detection for MediVan.

Features:
- YOLOv8 ONNX object detection
- OpenCV MOG2 fallback when YOLO is unavailable
- Filters irrelevant/static objects
- Only considers objects inside the driving corridor
- Uses confidence + proximity thresholds
- Temporal confirmation to reduce false STOP events
- Produces ObstacleResult objects compatible with the planner/main.py

Designed for Raspberry Pi 4.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import config as c
except Exception:
    c = None

try:
    from config import (
        ObstacleAction,
        ObstacleClass,
        ObstacleResult,
    )
except Exception:
    # Fallback definitions so the module can still be imported for testing.
    from enum import Enum

    class ObstacleAction(Enum):
        NOMINAL = 0
        SLOW = 1
        STOP = 2

    class ObstacleClass(Enum):
        UNKNOWN = 0
        PERSON = 1
        CART = 2
        FURNITURE = 3
        MEDICAL_EQUIPMENT = 4
        EQUIPMENT = 5

    class ObstacleResult:
        def __init__(
            self,
            bbox=(0, 0, 0, 0),
            area=0,
            proximity=0.0,
            action=ObstacleAction.NOMINAL,
            classification=ObstacleClass.UNKNOWN,
            confidence=0.0,
        ):
            self.bbox = bbox
            self.area = area
            self.proximity = proximity
            self.action = action
            self.classification = classification
            self.confidence = confidence

        @property
        def cls(self):
            """Legacy read-only alias; canonical field is classification."""
            return self.classification


# ============================================================
# SETTINGS
# ============================================================

# YOLO model location.
DEFAULT_MODEL_PATHS = [
    "assets/yolov8n.onnx",
    "models/yolov8n.onnx",
    "yolov8n.onnx",
    "models/yolo/yolov8n.onnx",
]

# YOLO input resolution.
YOLO_INPUT_SIZE = 320

# Minimum confidence for accepting a detection.
MIN_CONFIDENCE = 0.45

# Strong confidence needed for safety decisions.
SAFETY_CONFIDENCE = 0.65

# NMS threshold.
NMS_THRESHOLD = 0.45

# Camera driving corridor.
#
# Objects outside this horizontal region are normally ignored
# for navigation purposes.
CORRIDOR_LEFT = 0.30
CORRIDOR_RIGHT = 0.70

# Slightly wider corridor for very large objects.
WIDE_CORRIDOR_LEFT = 0.20
WIDE_CORRIDOR_RIGHT = 0.80

# Proximity thresholds.
#
# proximity is approximately:
# bottom of bounding box / image height
#
# Higher value = object appears closer.
SLOW_PROXIMITY = 0.62
STOP_PROXIMITY = 0.80

# Bounding-box area thresholds.
MIN_OBJECT_AREA = 0.005
STOP_AREA_RATIO = 0.28
SLOW_AREA_RATIO = 0.10

# Temporal confirmation.
#
# A single bad YOLO frame will not immediately generate STOP.
REQUIRED_STOP_CONFIRMATIONS = 3

# Number of previous frames kept for tracking.
HISTORY_LENGTH = 5

# Maximum number of detections.
MAX_DETECTIONS = 20


# ============================================================
# COCO CLASS NAMES
# ============================================================

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# ============================================================
# COCO -> PROJECT CLASS MAPPING
# ============================================================

def _get_project_class(name: str):
    """
    Convert a YOLO/COCO class into the project's ObstacleClass.
    """

    name = name.lower().strip()

    if name == "person":
        return ObstacleClass.PERSON

    # Bicycle/cart-like mobile objects.
    if name in {
        "bicycle",
        "suitcase",
    }:
        return ObstacleClass.CART

    # Medical/environmental objects that may physically block the robot.
    if name in {
        "bed",
        "bottle",
    }:
        return ObstacleClass.MEDICAL_EQUIPMENT

    # Static furniture.
    if name in {
        "chair",
        "couch",
        "bench",
        "dining table",
        "potted plant",
    }:
        return ObstacleClass.FURNITURE

    # Other equipment.
    if name in {
        "backpack",
        "handbag",
        "laptop",
        "tv",
        "book",
        "clock",
        "cell phone",
        "remote",
        "keyboard",
    }:
        return ObstacleClass.EQUIPMENT

    return ObstacleClass.UNKNOWN


# Classes that should be allowed to influence navigation.
NAVIGATION_CLASSES = {
    ObstacleClass.PERSON,
    ObstacleClass.CART,
    ObstacleClass.MEDICAL_EQUIPMENT,
    ObstacleClass.FURNITURE,
    ObstacleClass.EQUIPMENT,
}


# Classes that are normally static/non-blocking.
STATIC_CLASSES = set()


# ============================================================
# AI OBSTACLE DETECTOR
# ============================================================

class AIObstacleDetector:
    """
    Main obstacle detector.

    Usage:

        detector = AIObstacleDetector()

        obstacles = detector.detect(frame)

        for obstacle in obstacles:
            print(obstacle.classification)
            print(obstacle.action)

        detector.close()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_heuristic_fallback: bool = True,
    ):

        self.model_path = model_path
        self.use_heuristic_fallback = use_heuristic_fallback

        self.net = None
        self.model_loaded = False
        self.using_heuristic = False

        # Detection history.
        self.history = deque(maxlen=HISTORY_LENGTH)

        # Confirmation counters.
        self.confirmation_counter = defaultdict(int)

        # Previous detection positions.
        self.previous_objects = []

        # Statistics and inference-stride cache.
        self.frame_count = 0
        self.last_detection_time = 0.0
        self._cached_results = []
        self._last_inference_frame = -1

        # MOG2 fallback.
        self.bg_subtractor = None

        # Load YOLO.
        self._load_model()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    @property
    def mode(self) -> str:
        """Public detector mode used by the HUD and evaluation tools."""
        if self.model_loaded and self.net is not None:
            return "YOLO"
        if self.using_heuristic:
            return "HEURISTIC"
        return "NONE"

    def _find_model(self) -> Optional[str]:
        """Find the configured model relative to the project, not the CWD."""
        project_root = Path(__file__).resolve().parents[1]
        paths = []
        if self.model_path:
            paths.append(self.model_path)
        if c is not None:
            paths.append(getattr(c, "YOLO_MODEL_PATH", None))
        paths.extend(DEFAULT_MODEL_PATHS)

        seen = set()
        for candidate in paths:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            checks = [path] if path.is_absolute() else [project_root / path, path]
            for check in checks:
                full = str(check.resolve())
                if full in seen:
                    continue
                seen.add(full)
                if os.path.isfile(full):
                    return full
        return None

    def _load_model(self) -> None:
        """
        Load YOLO ONNX model.
        """

        model_path = self._find_model()

        if model_path is None:

            print(
                "[AI] YOLO model not found. "
                "Using OpenCV heuristic detector."
            )

            self._init_heuristic()
            return

        try:

            print(f"[AI] Loading YOLO model: {model_path}")

            self.net = cv2.dnn.readNetFromONNX(model_path)

            # Raspberry Pi CPU.
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.model_loaded = True
            self.using_heuristic = False

            print("[AI] YOLO model loaded successfully.")

        except Exception as e:

            print(f"[AI] Failed to load YOLO model: {e}")

            self.net = None
            self.model_loaded = False

            if self.use_heuristic_fallback:
                self._init_heuristic()

    # ========================================================
    # HEURISTIC FALLBACK
    # ========================================================

    def _init_heuristic(self) -> None:
        """
        Initialize OpenCV motion detector.
        """

        try:

            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300,
                varThreshold=32,
                detectShadows=False,
            )

            self.using_heuristic = True

            print("[AI] Heuristic motion detector enabled.")

        except Exception as e:

            print(f"[AI] Failed to initialize heuristic detector: {e}")

            self.bg_subtractor = None
            self.using_heuristic = False

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
            swapRB=True,
            crop=False,
        )

        scale_x = w / float(YOLO_INPUT_SIZE)
        scale_y = h / float(YOLO_INPUT_SIZE)

        return blob, scale_x, scale_y

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def _run_yolo(
        self,
        frame: np.ndarray,
    ) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:

        if self.net is None:
            return []

        blob, scale_x, scale_y = self._preprocess(frame)

        try:

            self.net.setInput(blob)

            outputs = self.net.forward()

        except Exception as e:

            print(f"[AI] YOLO inference error: {e}")
            return []

        # ----------------------------------------------------
        # Normalize output shape.
        #
        # Typical YOLOv8 ONNX:
        # (1, 84, 8400)
        #
        # We convert it to:
        # (8400, 84)
        # ----------------------------------------------------

        output = np.squeeze(outputs)

        if output.ndim != 2:
            print(f"[AI] Unexpected YOLO output shape: {outputs.shape}")
            return []

        if output.shape[0] < output.shape[1]:
            output = output.T

        detections = []

        boxes = []
        confidences = []
        class_ids = []

        h, w = frame.shape[:2]

        for row in output:

            if len(row) < 6:
                continue

            cx = float(row[0])
            cy = float(row[1])
            bw = float(row[2])
            bh = float(row[3])

            class_scores = row[4:]

            if len(class_scores) == 0:
                continue

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < MIN_CONFIDENCE:
                continue

            if class_id >= len(COCO_CLASSES):
                continue

            # YOLO coordinates are based on model input size.
            x_center = cx * scale_x
            y_center = cy * scale_y
            box_width = bw * scale_x
            box_height = bh * scale_y

            x = int(x_center - box_width / 2)
            y = int(y_center - box_height / 2)

            box_width = int(box_width)
            box_height = int(box_height)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            box_width = max(
                1,
                min(box_width, w - x),
            )

            box_height = max(
                1,
                min(box_height, h - y),
            )

            boxes.append(
                [
                    x,
                    y,
                    box_width,
                    box_height,
                ]
            )

            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        # NMS.
        try:

            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                MIN_CONFIDENCE,
                NMS_THRESHOLD,
            )

        except Exception as e:

            print(f"[AI] NMS error: {e}")
            return []

        if len(indices) == 0:
            return []

        indices = np.array(indices).reshape(-1)

        for i in indices:

            i = int(i)

            detections.append(
                (
                    class_ids[i],
                    confidences[i],
                    tuple(boxes[i]),
                )
            )

            if len(detections) >= MAX_DETECTIONS:
                break

        return detections

    # ========================================================
    # GEOMETRY
    # ========================================================

    def _bbox_center(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:

        x, y, w, h = bbox

        return (
            x + w / 2.0,
            y + h / 2.0,
        )

    def _bbox_area_ratio(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape,
    ) -> float:

        _, _, bw, bh = bbox

        frame_h, frame_w = frame_shape[:2]

        if frame_w <= 0 or frame_h <= 0:
            return 0.0

        return (
            float(bw * bh)
            / float(frame_w * frame_h)
        )

    def _proximity(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape,
    ) -> float:

        frame_h, frame_w = frame_shape[:2]

        if frame_h <= 0:
            return 0.0

        x, y, w, h = bbox

        bottom_y = y + h

        proximity = bottom_y / float(frame_h)

        return max(
            0.0,
            min(1.0, proximity),
        )

    # ========================================================
    # DRIVING CORRIDOR
    # ========================================================

    def _is_in_driving_corridor(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape,
    ) -> bool:
        """
        Determine whether an object is actually in front
        of the MediVan.

        Objects at the extreme left/right of the image are
        ignored for navigation.
        """

        frame_h, frame_w = frame_shape[:2]

        if frame_w <= 0:
            return False

        x, y, w, h = bbox

        center_x = x + w / 2.0

        normalized_x = center_x / float(frame_w)

        # Normal driving corridor.
        if CORRIDOR_LEFT <= normalized_x <= CORRIDOR_RIGHT:
            return True

        # Large object can extend into the path.
        object_left = x / float(frame_w)
        object_right = (x + w) / float(frame_w)

        overlaps_center = (
            object_left <= 0.50 <= object_right
        )

        if overlaps_center:

            # Only allow this if the object is reasonably large.
            area_ratio = self._bbox_area_ratio(
                bbox,
                frame_shape,
            )

            if area_ratio >= SLOW_AREA_RATIO:
                return True

        # Wider corridor only for very large objects.
        if area_ratio if "area_ratio" in locals() else False:
            pass

        return False

    # ========================================================
    # OBJECT RELEVANCE
    # ========================================================

    def _is_relevant_class(
        self,
        obstacle_class,
    ) -> bool:

        return obstacle_class in NAVIGATION_CLASSES

    # ========================================================
    # ACTION CALCULATION
    # ========================================================

    def _calculate_action(
        self,
        obstacle_class,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        proximity: float,
        frame_shape,
    ):
        """
        Decide NOMINAL / SLOW / STOP.
        """

        # ----------------------------------------------------
        # Static objects never generate a navigation STOP.
        # ----------------------------------------------------

        if obstacle_class in STATIC_CLASSES:
            return ObstacleAction.NOMINAL

        # Unknown objects are ignored for safety decisions.
        if obstacle_class == ObstacleClass.UNKNOWN:
            return ObstacleAction.NOMINAL

        # Only relevant mobile/physical objects continue.
        if obstacle_class not in NAVIGATION_CLASSES:
            return ObstacleAction.NOMINAL

        # ----------------------------------------------------
        # Confidence gate.
        # ----------------------------------------------------

        if confidence < MIN_CONFIDENCE:
            return ObstacleAction.NOMINAL

        # ----------------------------------------------------
        # Driving corridor gate.
        # ----------------------------------------------------

        if not self._is_in_driving_corridor(
            bbox,
            frame_shape,
        ):
            return ObstacleAction.NOMINAL

        # ----------------------------------------------------
        # Size.
        # ----------------------------------------------------

        area_ratio = self._bbox_area_ratio(
            bbox,
            frame_shape,
        )

        # Very tiny object.
        if area_ratio < MIN_OBJECT_AREA:
            return ObstacleAction.NOMINAL

        # ----------------------------------------------------
        # Low confidence objects should not stop robot.
        # ----------------------------------------------------

        if confidence < SAFETY_CONFIDENCE:

            if proximity >= SLOW_PROXIMITY:
                return ObstacleAction.SLOW

            return ObstacleAction.NOMINAL

        # ----------------------------------------------------
        # STOP condition.
        # ----------------------------------------------------

        stop_condition = (
            proximity >= STOP_PROXIMITY
            or area_ratio >= STOP_AREA_RATIO
        )

        if stop_condition:

            # Require temporal confirmation.
            return ObstacleAction.STOP

        # ----------------------------------------------------
        # SLOW condition.
        # ----------------------------------------------------

        slow_condition = (
            proximity >= SLOW_PROXIMITY
            or area_ratio >= SLOW_AREA_RATIO
        )

        if slow_condition:
            return ObstacleAction.SLOW

        return ObstacleAction.NOMINAL

    # ========================================================
    # TEMPORAL CONFIRMATION
    # ========================================================

    def _object_key(
        self,
        obstacle_class,
        bbox,
    ):

        x, y, w, h = bbox

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Quantize location.
        gx = int(cx / 40)
        gy = int(cy / 40)

        return (
            str(obstacle_class),
            gx,
            gy,
        )

    def _apply_temporal_confirmation(
        self,
        obstacle_class,
        bbox,
        action,
    ):

        key = self._object_key(
            obstacle_class,
            bbox,
        )

        if action == ObstacleAction.STOP:

            self.confirmation_counter[key] += 1

            if (
                self.confirmation_counter[key]
                >= REQUIRED_STOP_CONFIRMATIONS
            ):
                return ObstacleAction.STOP

            # First frame:
            # slow instead of immediately stopping.
            return ObstacleAction.SLOW

        # Reset nearby confirmation when object no longer
        # qualifies for STOP.
        self.confirmation_counter[key] = max(
            0,
            self.confirmation_counter[key] - 1,
        )

        return action

    # ========================================================
    # CONVERT YOLO DETECTIONS
    # ========================================================

    def _convert_detections(
        self,
        raw_detections,
        frame,
    ) -> List[ObstacleResult]:

        results = []

        for class_id, confidence, bbox in raw_detections:

            if class_id < 0 or class_id >= len(COCO_CLASSES):
                continue

            name = COCO_CLASSES[class_id]

            obstacle_class = _get_project_class(name)

            # ------------------------------------------------
            # Ignore classes that are not part of navigation.
            # We completely omit them from results.
            # ------------------------------------------------

            if (
                obstacle_class == ObstacleClass.UNKNOWN
                or obstacle_class in STATIC_CLASSES
            ):
                continue

            proximity = self._proximity(
                bbox,
                frame.shape,
            )

            action = self._calculate_action(
                obstacle_class,
                confidence,
                bbox,
                proximity,
                frame.shape,
            )

            action = self._apply_temporal_confirmation(
                obstacle_class,
                bbox,
                action,
            )

            result = ObstacleResult(
                classification=obstacle_class,
                confidence=float(confidence),
                bbox=bbox,
                proximity=float(proximity),
                action=action,
            )

            # Attach extra information dynamically.
            # This does not break the existing ObstacleResult.
            try:
                result.label = name
                result.area_ratio = self._bbox_area_ratio(
                    bbox,
                    frame.shape,
                )
                result.in_corridor = self._is_in_driving_corridor(
                    bbox,
                    frame.shape,
                )
            except Exception:
                pass

            results.append(result)

        # Sort by danger.
        results.sort(
            key=lambda obj: (
                obj.action.value
                if hasattr(obj.action, "value")
                else 0,
                obj.proximity,
                obj.confidence,
            ),
            reverse=True,
        )

        return results[:MAX_DETECTIONS]

    # ========================================================
    # HEURISTIC MOTION DETECTION
    # ========================================================

    def _detect_heuristic(
        self,
        frame: np.ndarray,
    ) -> List[ObstacleResult]:

        if self.bg_subtractor is None:
            return []

        mask = self.bg_subtractor.apply(frame)

        # Remove noise.
        kernel = np.ones(
            (5, 5),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        results = []

        frame_h, frame_w = frame.shape[:2]

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            bbox = (
                int(x),
                int(y),
                int(w),
                int(h),
            )

            area_ratio = (
                area
                / float(frame_w * frame_h)
            )

            # Only motion in driving corridor.
            if not self._is_in_driving_corridor(
                bbox,
                frame.shape,
            ):
                continue

            proximity = self._proximity(
                bbox,
                frame.shape,
            )

            # Heuristic detector cannot know exact object class.
            obstacle_class = ObstacleClass.PERSON

            confidence = 0.60

            if (
                proximity >= STOP_PROXIMITY
                and area_ratio >= 0.08
            ):
                action = ObstacleAction.SLOW
            elif proximity >= SLOW_PROXIMITY:
                action = ObstacleAction.SLOW
            else:
                action = ObstacleAction.NOMINAL

            result = ObstacleResult(
                classification=obstacle_class,
                confidence=confidence,
                bbox=bbox,
                proximity=proximity,
                action=action,
            )

            try:
                result.label = "motion"
                result.area_ratio = area_ratio
                result.in_corridor = True
            except Exception:
                pass

            results.append(result)

        results.sort(
            key=lambda obj: obj.proximity,
            reverse=True,
        )

        return results[:MAX_DETECTIONS]

    # ========================================================
    # MAIN DETECTION FUNCTION
    # ========================================================

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[ObstacleResult]:
        """
        Detect obstacles in a camera frame.

        Parameters
        ----------
        frame:
            BGR OpenCV image.

        Returns
        -------
        List[ObstacleResult]
        """

        self.frame_count += 1

        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            return []

        if frame.size == 0:
            return []

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        if self.model_loaded and self.net is not None:

            stride = max(1, int(getattr(c, "YOLO_SKIP_FRAMES", 1))) if c is not None else 1
            run_inference = (
                self._last_inference_frame < 0
                or self.frame_count - self._last_inference_frame >= stride
            )
            if run_inference:
                raw_detections = self._run_yolo(frame)
                obstacles = self._convert_detections(raw_detections, frame)
                self._cached_results = list(obstacles)
                self._last_inference_frame = self.frame_count
                self.last_detection_time = time.time()
            else:
                # Retain the last detection set for at most stride-1 frames.
                # In particular, a detected STOP is not dropped between CPU
                # inference passes.
                obstacles = list(self._cached_results)

        # ----------------------------------------------------
        # Heuristic fallback
        # ----------------------------------------------------

        elif self.using_heuristic:

            obstacles = self._detect_heuristic(frame)

        else:

            obstacles = []

        # Store history.
        self.history.append(obstacles)

        if not (self.model_loaded and self.net is not None):
            self.last_detection_time = time.time()

        return obstacles

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def draw_detections(
        self,
        frame: np.ndarray,
        obstacles: List[ObstacleResult],
    ) -> np.ndarray:
        """
        Draw detection boxes and navigation status.

        This is optional and can be called by camera_sim.py.
        """

        if frame is None:
            return frame

        output = frame.copy()

        frame_h, frame_w = output.shape[:2]

        # Driving corridor.
        left_x = int(frame_w * CORRIDOR_LEFT)
        right_x = int(frame_w * CORRIDOR_RIGHT)

        cv2.line(
            output,
            (left_x, 0),
            (left_x, frame_h),
            (180, 180, 180),
            1,
        )

        cv2.line(
            output,
            (right_x, 0),
            (right_x, frame_h),
            (180, 180, 180),
            1,
        )

        for obstacle in obstacles:

            x, y, w, h = obstacle.bbox

            # Determine display color.
            if obstacle.action == ObstacleAction.STOP:
                color = (0, 0, 255)
                action_text = "STOP"

            elif obstacle.action == ObstacleAction.SLOW:
                color = (0, 165, 255)
                action_text = "SLOW"

            else:
                color = (0, 255, 0)
                action_text = "OK"

            label = getattr(
                obstacle,
                "label",
                str(obstacle.classification),
            )

            text = (
                f"{label} "
                f"{obstacle.confidence:.2f} "
                f"{action_text}"
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

            text_y = max(
                20,
                y - 8,
            )

            cv2.putText(
                output,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Status.
        if self.model_loaded:
            mode_text = "YOLO"
        elif self.using_heuristic:
            mode_text = "MOTION"
        else:
            mode_text = "NONE"

        cv2.putText(
            output,
            f"AI: {mode_text}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    # ========================================================
    # RESET
    # ========================================================

    def reset_tracking(self) -> None:
        """
        Reset temporal detection history.
        """

        self.history.clear()
        self.confirmation_counter.clear()
        self.previous_objects.clear()
        self._cached_results.clear()
        self._last_inference_frame = -1

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Release detector resources.
        """

        self.net = None
        self.bg_subtractor = None

        self.history.clear()
        self.confirmation_counter.clear()
        self.previous_objects.clear()
        self._cached_results.clear()
        self._last_inference_frame = -1


# ============================================================
# SIMPLE TEST
# ============================================================

def main():
    """
    Standalone camera test.

    Run:

        python3 ai_obstacle_detector.py

    Press Q to exit.
    """

    print("=" * 60)
    print("MediVan AI Obstacle Detector Test")
    print("=" * 60)

    detector = AIObstacleDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("[AI TEST] Cannot open camera.")

        detector.close()
        return

    print("[AI TEST] Camera started.")
    print("[AI TEST] Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[AI TEST] Failed to read frame.")
            break

        obstacles = detector.detect(frame)

        for obstacle in obstacles:

            label = getattr(
                obstacle,
                "label",
                str(obstacle.classification),
            )

            print(
                f"[AI] "
                f"{label} | "
                f"confidence={obstacle.confidence:.2f} | "
                f"proximity={obstacle.proximity:.2f} | "
                f"action={obstacle.action}"
            )

        display = detector.draw_detections(
            frame,
            obstacles,
        )

        cv2.imshow(
            "MediVan AI Obstacle Detector",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    detector.close()

    print("[AI TEST] Detector stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
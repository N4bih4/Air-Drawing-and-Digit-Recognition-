import cv2
import mediapipe as mp
import math
import urllib.request
import os
import time

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initializes the HandDetector with MediaPipe Tasks Vision API.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Download model if not exists
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe Hand Landmarker model to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete!")

        # Initialize Tasks API
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=self.running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.hands = HandLandmarker.create_from_options(options)
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        self.results = None
        self.last_timestamp_ms = 0

        # Standard MediaPipe Hand Connections for custom drawing
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
        ]

    def find_hands(self, img, draw=True):
        """
        Finds hands in a video frame and optionally draws landmarks.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # In VIDEO mode, timestamp must be monotonically increasing.
        if self.running_mode == mp.tasks.vision.RunningMode.VIDEO:
            current_time_ms = int(time.time() * 1000)
            if current_time_ms <= self.last_timestamp_ms:
                current_time_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = current_time_ms
            
            self.results = self.hands.detect_for_video(mp_image, current_time_ms)
        else:
            self.results = self.hands.detect(mp_image)

        if self.results and getattr(self.results, 'hand_landmarks', None):
            for hand_landmarks in self.results.hand_landmarks:
                if draw:
                    self._draw_landmarks(img, hand_landmarks)
        return img
        
    def _draw_landmarks(self, img, landmarks):
        """Draws hand skeleton (used when draw=True in find_hands)."""
        self._draw_landmarks_styled(img, landmarks)

    def _draw_landmarks_styled(self, img, landmarks, y_min=0, y_max=None, x_margin=0):
        """Purple skeleton + highlighted fingertips; optional canvas clipping."""
        h, w = img.shape[:2]
        if y_max is None:
            y_max = h

        line_color = (210, 85, 195)
        joint_color = (255, 255, 255)
        tip_colors = {4: (0, 200, 255), 8: (80, 230, 80), 12: (255, 180, 80)}

        for start_idx, end_idx in self.HAND_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
            x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
            if y_min <= y1 <= y_max and y_min <= y2 <= y_max:
                cv2.line(img, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)

        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if not (y_min <= cy <= y_max and x_margin <= cx <= w - x_margin):
                continue
            color = tip_colors.get(idx, joint_color)
            radius = 7 if idx in tip_colors else 4
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
            if idx in tip_colors:
                cv2.circle(img, (cx, cy), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_hand_overlay(self, img, canvas_top, canvas_bottom, x_margin=10):
        """Draw hand tracking overlay on the canvas area only."""
        if not self.results or not getattr(self.results, "hand_landmarks", None):
            return False
        for hand_landmarks in self.results.hand_landmarks:
            self._draw_landmarks_styled(
                img, hand_landmarks, y_min=canvas_top, y_max=canvas_bottom, x_margin=x_margin
            )
        return True

    def find_position(self, img, hand_no=0, draw=True):
        """
        Returns a list of landmark positions for the specified hand.
        """
        self.lm_list = []
        if self.results and getattr(self.results, 'hand_landmarks', None):
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        """
        Checks which fingers are up.
        Returns a list of 5 elements (0 or 1) representing [Thumb, Index, Middle, Ring, Pinky].
        """
        fingers = []
        if len(self.lm_list) == 0:
            return []

        # Thumb
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

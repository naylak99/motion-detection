import cv2
import numpy as np
from core.frame_processor import FrameProcessor  # Fixed import
from utils.logger import Logger  # Fixed import

class BasicMotionDetector:
    def __init__(self, config):
        self.config = config
        self.frame_processor = FrameProcessor(config)
        self.background_frame = None
        self.logger = Logger()

    def detect_motion(self, frame):
        # Resize and preprocess frame
        frame = self.frame_processor.resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.frame_processor.apply_gaussian_blur(gray)

        # Initialize background
        if self.background_frame is None:
            self.background_frame = gray
            return frame, None, None, None

        # Calculate frame delta
        frame_delta = self.frame_processor.fast_frame_delta(self.background_frame, gray)
        thresh = self.frame_processor.get_binary_threshold(frame_delta)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < self.config['min_area']:
                continue
                
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            self.logger.log_motion_event()

        return frame, gray, thresh, frame_delta
import cv2
import numpy as np
from collections import deque
from core.frame_processor import FrameProcessor  # Fixed import
from utils.logger import Logger  # Fixed import

class AdvancedMotionDetector:
    def __init__(self, config):
        self.config = config
        self.frame_processor = FrameProcessor(config)
        self.motion_buffer = deque(maxlen=config['buffer_size'])
        self.background_buffer = deque(maxlen=30)
        self.logger = Logger()

    def update_buffers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.frame_processor.apply_gaussian_blur(gray)
        
        self.motion_buffer.append(gray)
        if len(self.motion_buffer) < self.motion_buffer.maxlen:
            return None
            
        self.background_buffer.append(gray)
        return gray

    def get_average_background(self):
        if len(self.background_buffer) < 1:
            return None
            
        return np.mean(self.background_buffer, axis=0).astype(np.uint8)

    def detect_motion(self, frame):
        # Preprocess frame
        frame = self.frame_processor.resize_frame(frame)
        current_frame = self.update_buffers(frame)
        
        if current_frame is None:
            return frame, None, None, None

        # Get average background
        avg_background = self.get_average_background()
        if avg_background is None:
            return frame, None, None, None

        # Calculate weighted frame delta
        frame_delta = self.frame_processor.fast_frame_delta(avg_background, current_frame)
        thresh = self.frame_processor.get_binary_threshold(frame_delta)

        # Find and process contours with noise filtering
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config['min_area']:
                continue
                
            # Additional noise filtering
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.1:  # Filter out non-circular noise
                continue

            valid_contours.append(contour)
            motion_detected = True

        # Draw contours
        for contour in valid_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            self.logger.log_motion_event()

        return frame, current_frame, thresh, frame_delta
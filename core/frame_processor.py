import cv2
import numpy as np
from numba import jit

# Separate standalone function for Numba JIT compilation
@jit(nopython=True)
def compute_frame_delta(frame1, frame2):
    return np.absolute(frame1.astype(np.float32) - frame2.astype(np.float32))

class FrameProcessor:
    def __init__(self, config):
        self.config = config

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.config['resize_width'], 
                                self.config['resize_height']))

    def apply_gaussian_blur(self, frame):
        return cv2.GaussianBlur(frame, self.config['blur_kernel_size'], 0)

    def fast_frame_delta(self, frame1, frame2):
        # Convert frames to numpy arrays and ensure they're the right type
        f1 = np.array(frame1, dtype=np.float32)
        f2 = np.array(frame2, dtype=np.float32)
        # Use the JIT-compiled function
        delta = compute_frame_delta(f1, f2)
        return delta.astype(np.uint8)

    def get_binary_threshold(self, frame_delta):
        _, thresh = cv2.threshold(frame_delta, self.config['motion_threshold'], 
                                255, cv2.THRESH_BINARY)
        return cv2.dilate(thresh, None, iterations=2)
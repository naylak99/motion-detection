import cv2

class VisualizationManager:
    @staticmethod
    def show_frames(color_frame, gray_frame, threshold_frame, delta_frame):
        frames = {
            'Color Frame': color_frame,
            'Gray Frame': gray_frame,
            'Threshold Frame': threshold_frame,
            'Delta Frame': delta_frame
        }
        
        for name, frame in frames.items():
            if frame is not None:
                cv2.imshow(name, frame)

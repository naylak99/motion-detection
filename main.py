import cv2
from config.settings import CONFIG
from core.advanced_detector import AdvancedMotionDetector
from utils.visualization import VisualizationManager

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    
    # Initialize detector and visualizer
    detector = AdvancedMotionDetector(CONFIG)
    visualizer = VisualizationManager()

    print("Motion detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            # Detect motion
            color_frame, gray_frame, threshold_frame, delta_frame = detector.detect_motion(frame)
            
            # Display results
            visualizer.show_frames(color_frame, gray_frame, threshold_frame, delta_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
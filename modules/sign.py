import cv2
import os
import time
from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Load the trained sign language model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best.pt")
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize FPS calculation
prev_time = 0

try:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model.track(frame, persist=True, conf=0.65)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("Hand Sign Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
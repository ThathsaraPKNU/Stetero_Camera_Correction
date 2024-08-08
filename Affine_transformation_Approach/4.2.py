#Overlap realtime view with correction using affine transformation matrix
#Output view reduce
#recording when press s

from picamera2 import Picamera2, Preview
from time import sleep
import cv2
import numpy as np
import os

class Camera:
    def __init__(self, camera_id, resolution=(1640, 1232), frame_rate=25):
        self.cam = Picamera2(camera_id)
        config = self.cam.create_preview_configuration(main={"size": resolution, "format": "RGB888"})
        config["controls"]["FrameDurationLimits"] = (1000000 // frame_rate, 1000000 // frame_rate)
        self.cam.configure(config)
        self.cam.start()

    def capture(self):
        frame = self.cam.capture_array()
        return frame

if __name__ == "__main__":
    # Initialize cameras with the specified resolution and frame rate
    cam0 = Camera(camera_id=0, resolution=(1640, 1232), frame_rate=25)
    cam1 = Camera(camera_id=1, resolution=(1640, 1232), frame_rate=25)
    
    # Define the affine transformation matrix from your earlier result
    affine_matrix = np.array([[1.02591214, -0.0559109993, 126.231029],
                              [0.0559109993, 1.02591214, -90.5332862]])

    display_size = (640, 480)  # Size for displaying the video feed
    is_recording = False
    out = None

    while True:
        # Capture frames from both cameras
        frame0 = cam0.capture()
        frame1 = cam1.capture()
        
        # Apply the affine transformation to frame1
        frame1_transformed = cv2.warpAffine(frame1, affine_matrix, (frame1.shape[1], frame1.shape[0]))
        
        # Create a mask for the transformed frame from camera 1
        mask = cv2.threshold(cv2.cvtColor(frame1_transformed, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        
        # Apply the mask to camera 1's transformed frame
        frame1_with_mask = cv2.bitwise_and(frame1_transformed, frame1_transformed, mask=mask.astype(np.uint8))
        
        # Create an empty canvas for merging
        merged_frame = np.copy(frame0)
        
        # Overlay the masked and transformed frame from camera 1 onto the frame from camera 0
        merged_frame = cv2.addWeighted(merged_frame, 0.5, frame1_with_mask, 0.5, 0)
        
        # Resize the merged frame to the display size
        merged_frame_resized = cv2.resize(merged_frame, display_size)
        
        # Display the merged frame
        cv2.imshow("Merged Frames", merged_frame_resized)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            if not is_recording:
                # Start recording
                video_filename = os.path.join("/home/lab902/Videos", "recording.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(video_filename, fourcc, 25.0, display_size)
                is_recording = True
                print(f"Recording started: {video_filename}")
            else:
                # Stop recording
                is_recording = False
                out.release()
                out = None
                print("Recording stopped")

        if is_recording and out is not None:
            # Write the frame to the video file
            out.write(merged_frame_resized)

    # Release resources
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

#Overlap realtime view with correction using affine transformation matrix

from picamera2 import Picamera2, Preview
from time import sleep
import cv2
import numpy as np

class Camera:
    def __init__(self, camera_id):
        self.cam = Picamera2(camera_id)
        self.cam.preview_configuration.main.size = (1640, 1232)
        self.cam.preview_configuration.main.format = "RGB888"
        self.cam.preview_configuration.align()
        self.cam.configure("preview")
        self.cam.start()

    def capture(self):
        frame = self.cam.capture_array()
        return frame

if __name__ == "__main__":
    # Initialize cameras
    cam0 = Camera(camera_id=0)
    cam1 = Camera(camera_id=1)
    
    # Define the affine transformation matrix from your earlier result
    affine_matrix = np.array([[1.02591214, -0.0559109993, 126.231029],
                              [0.0559109993, 1.02591214, -90.5332862]])

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
        
        # Display the merged frame
        cv2.imshow("Merged Frames", merged_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    # Release resources
    cv2.destroyAllWindows()

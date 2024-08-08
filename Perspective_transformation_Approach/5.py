#Final GUI with apply Perspective transformation matrix to cam 1

import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
from picamera2 import Picamera2
import numpy as np

recording = False
cameras_open = False  # Flag to control the camera loop
cameras_thread = None  # Thread for the camera loop
resolution = (1640, 1232)  # Default resolution
frame_rate = 25  # Target frame rate
display_size = (640, 480)  # Size for displaying the video feed

# Perspective Transformation Matrix for camera 1
perspective_matrix_cam1 = np.array([
    [ 9.17643781e-01, -1.06331927e-01,  2.06641642e+02],
    [ 4.32354238e-02,  9.14868950e-01, -5.08135293e+01],
    [-2.72857162e-05, -5.10638853e-05,  1.00000000e+00]
])

def open_cameras():
    global cameras_open, cameras_thread
    try:
        if not cameras_open:
            # Create camera instances
            global cam1, cam2
            cam1 = Camera(camera_index=0)
            cam2 = Camera(camera_index=1, apply_perspective=True, perspective_matrix=perspective_matrix_cam1)
            
            cameras_open = True
            cameras_thread = threading.Thread(target=camera_loop)
            cameras_thread.start()
    except Exception as e:
        print(f"Error opening cameras: {e}")

def camera_loop():
    global cameras_open
    while cameras_open:
        frame1 = cam1.capture()
        frame2 = cam2.capture()

        # Resize frames to display_size
        frame1_display = cv2.resize(frame1, display_size)
        frame2_display = cv2.resize(frame2, display_size)

        cv2.imshow('Cam 0', frame1_display)
        cv2.imshow('Cam 1', frame2_display)

        if cv2.waitKey(1) & 0xFF == ord('q') or not cameras_open:
            break
    cv2.destroyAllWindows()

def record_cameras():
    global recording
    recording = True
    record_thread = threading.Thread(target=record_loop)
    record_thread.start()
    recording_status_label.config(text="Recording...")

def record_loop():
    try:
        global recording, resolution, frame_rate
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out0 = cv2.VideoWriter('/home/lab902/Videos/V3.avi', fourcc, frame_rate, resolution)
        out1 = cv2.VideoWriter('/home/lab902/Videos/V4.avi', fourcc, frame_rate, resolution)
        
        while recording:
            start_time = time.time()  # Record the start time of frame processing
            frame0 = cam1.capture()
            frame1 = cam2.capture()

            # Write the RGB frame directly to the video file
            out0.write(frame0)
            out1.write(frame1)
            
            # Calculate the actual time taken to process the frame
            processing_time = time.time() - start_time
            
            # Calculate the sleep time to maintain the target frame rate
            sleep_time = 1.0 / frame_rate - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        out0.release()
        out1.release()
    except Exception as e:
        print(f"Error recording: {e}")

def stop_recording():
    global recording
    recording = False
    recording_status_label.config(text="Recording stopped.")

def release_cameras():
    global cameras_open, cameras_thread
    cameras_open = False
    if cameras_thread is not None:
        cameras_thread.join()
    if cam1:
        cam1.release()
    if cam2:
        cam2.release()
    cv2.destroyAllWindows()

def open_cameras_thread():
    open_cameras()

def close_cameras_and_stop_recording():
    stop_recording()
    release_cameras()

class Camera:
    def __init__(self, camera_index=0, apply_perspective=False, perspective_matrix=None):
        self.cam = Picamera2(camera_num=camera_index)
        self.cam.preview_configuration.main.size = resolution
        self.cam.preview_configuration.main.format = "RGB888"
        self.cam.preview_configuration.align()
        self.cam.configure("preview")
        self.cam.start()
        self.apply_perspective = apply_perspective
        self.perspective_matrix = perspective_matrix

    def capture(self):
        frame = self.cam.capture_array()
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
        if self.apply_perspective and self.perspective_matrix is not None:
            frame = cv2.warpPerspective(frame, self.perspective_matrix, resolution)
        return frame

    def release(self):
        self.cam.stop()

TN = tk.Tk()
width, height = 300, 200
TN.geometry(f'{width}x{height}')
TN.title("Record Cameras")
TN.resizable(False, False)

open_button = ttk.Button(TN, text='Open Cameras', command=open_cameras_thread)
open_button.pack()

record_button = ttk.Button(TN, text='Record', command=record_cameras)
record_button.pack()

recording_status_label = ttk.Label(TN, text="")
recording_status_label.pack()

stop_button = ttk.Button(TN, text='Stop Record', command=stop_recording)
stop_button.pack()

off_button = ttk.Button(TN, text='Off Cameras', command=close_cameras_and_stop_recording)
off_button.pack()

TN.mainloop()

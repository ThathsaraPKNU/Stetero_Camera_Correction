#Prepaire Aligned image and print Perspective Transformation Matrix

import cv2
import numpy as np

# Define the file paths
image1_path = '/home/lab902/Documents/Test Perspective Transformation/A.png'
image2_path = '/home/lab902/Documents/Test Perspective Transformation/B.png'

# Load the images from both cameras
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Check if the images are loaded correctly
if image1 is None:
    print(f"Error: Unable to load image from {image1_path}")
    exit(1)
if image2 is None:
    print(f"Error: Unable to load image from {image2_path}")
    exit(1)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate perspective transformation matrix (homography)
matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# Print the perspective transformation matrix
print("Perspective Transformation Matrix:")
print(matrix)

# Apply the perspective transformation to the second image
height, width = image1.shape[:2]
aligned_image = cv2.warpPerspective(image2, matrix, (width, height))

# Save or display the aligned image
cv2.imwrite('aligned_image.png', aligned_image)

print("Image alignment completed successfully. The aligned image has been saved.")

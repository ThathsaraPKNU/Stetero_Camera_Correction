#Prepaire Aligned image and print Homography Matrix

import cv2
import numpy as np

# Define the file paths
image1_path = '/home/lab902/Documents/Test 8.2024/A.png'
image2_path = '/home/lab902/Documents/Test 8.2024/B.png'

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

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Use FLANN-based matcher for feature matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Store all the good matches as per Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract the matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Estimate the homography matrix using RANSAC
matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# Print the homography matrix
print("Homography Matrix:")
print(matrix)

# Apply the homography transformation to the second image
height, width = image1.shape[:2]
aligned_image = cv2.warpPerspective(image2, matrix, (width, height))

# Save or display the aligned image
cv2.imwrite('aligned_image.png', aligned_image)

print("Image alignment completed successfully. The aligned image has been saved.")

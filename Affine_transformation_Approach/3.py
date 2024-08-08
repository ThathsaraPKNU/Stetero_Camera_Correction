#Prepaire Aligned image and print Affine Transformation Matrix

import cv2
import numpy as np

# Define the file paths
image1_path = '/home/lab902/Documents/Test 5.8.2024/C.png'
image2_path = '/home/lab902/Documents/Test 5.8.2024/D.png'

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

# Estimate affine transformation matrix
matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

# Print the affine transformation matrix
print("Affine Transformation Matrix:")
print(matrix)

# Apply the affine transformation to the second image
aligned_image = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))

# Save or display the aligned image
cv2.imwrite('aligned_image.png', aligned_image)

print("Image alignment completed successfully. The aligned image has been saved.")

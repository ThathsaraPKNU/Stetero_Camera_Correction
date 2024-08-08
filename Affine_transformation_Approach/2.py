#Prepaire Aligned image and Print the coordinates of 1st camera (100, 150) in the second camera correspond

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

# Check if descriptors are found
if descriptors1 is None or descriptors2 is None:
    print("Error: No descriptors found in one of the images.")
    exit(1)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Check if there are enough matches
if len(matches) < 4:
    print(f"Error: Not enough matches found ({len(matches)}). At least 4 matches are required.")
    exit(1)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate affine transformation matrix
matrix, mask = cv2.estimateAffinePartial2D(points2, points1)

# Apply the affine transformation to the second image
aligned_image = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))

# Save the aligned image
output_path = '/home/lab902/Documents/Images/aligned_image.png'
cv2.imwrite(output_path, aligned_image)

print(f"Image alignment completed successfully. The aligned image has been saved to {output_path}.")

# Function to transform pixel coordinates from the second camera to the first camera
def transform_coordinates(x, y, matrix):
    point = np.array([x, y, 1]).reshape(3, 1)  # Create a column vector
    transformed_point = np.dot(matrix, point)  # Apply the affine transformation
    transformed_x = transformed_point[0, 0]
    transformed_y = transformed_point[1, 0]
    return transformed_x, transformed_y

# Example usage: Transform a pixel coordinate (x2, y2) from the second camera
x2, y2 = 100, 150  # Example coordinates from the second camera
x1, y1 = transform_coordinates(x2, y2, matrix)
print(f"The coordinates ({x2}, {y2}) in the second camera correspond to ({x1}, {y1}) in the first camera.")

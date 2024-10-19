import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('inputimg.webp', cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not read image.")
    exit()

# Apply Sobel filter on the X axis
sobelx = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)

# Apply Sobel filter on the Y axis
sobely = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# Combine the Sobel X and Y images using magnitude
sobel_combined = cv2.magnitude(sobelx, sobely)

# Apply Canny edge detector
edges = cv2.Canny(image=image, threshold1=100, threshold2=200)


# List of titles and images
titles = ['Original Image', 'Sobel X', 'Sobel Y', 'Sobel Combined', 'Canny Edges']
images = [image, sobelx, sobely, sobel_combined, edges]

# Set up the matplotlib figure
plt.figure(figsize=(10, 7))

# Loop through images and display them
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# Save the images
cv2.imwrite('output_sobelx.jpg', sobelx)
cv2.imwrite('output_sobely.jpg', sobely)
cv2.imwrite('output_sobel_combined.jpg', sobel_combined)
cv2.imwrite('output_canny_edges.jpg', edges)

import pandas as pd
import numpy as np

from glob import glob

import cv2
import matplotlib.pylab as plt

plt.style.use('ggplot')
def apply_spatial_filters(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
    
    # Convert to RGB (OpenCV loads in BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Gaussian Blur (Smoothing)
    # (5, 5) is the kernel size, 0 means sigma is calculated from kernel size
    gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    # 3. Median Filter (Great for salt-and-pepper noise)
    # 5 is the aperture linear size (must be odd)
    median = cv2.medianBlur(img_rgb, 5)

    # 4. Sharpening (Using a custom kernel)
    # Define a 3x3 sharpening kernel
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    
    # Apply the kernel using filter2D
    sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpening)

    # 5. Visualization
    titles = ['Original', 'Gaussian Blur', 'Median Filter', 'Sharpened']
    images = [img_rgb, gaussian, median, sharpened]

    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the function
# apply_spatial_filters('your_image.jpg')
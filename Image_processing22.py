import os
import cv2
import numpy as np
import zipfile
import random
import pandas as pd
import matplotlib.pylab as plt
from google.colab import files
from glob import glob

# --- 1. UPLOAD AND EXTRACT ---
print("Step 1: Please upload your zip file.")
uploaded = files.upload() # This will prompt you to select your zip file

zip_path = "emotion detection project.v1i.folder.zip"  
extract_path = "dataset"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset Extracted ✅")
else:
    print("Error: Zip file not found. Please ensure the name matches.")

# --- 2. PREPROCESSING CONFIGURATION ---
IMG_SIZE = 224
plt.style.use('ggplot')

def preprocess_folder(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} (Folder not found)")
        return

    os.makedirs(output_path, exist_ok=True)

    for class_name in os.listdir(input_path):
        class_input = os.path.join(input_path, class_name)
        class_output = os.path.join(output_path, class_name)

        if not os.path.isdir(class_input):
            continue

        os.makedirs(class_output, exist_ok=True)

        for img_name in os.listdir(class_input):
            img_path = os.path.join(class_input, img_name)
            save_path = os.path.join(class_output, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Processing Pipeline
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            # Normalization/Renormalization for saving
            normalized = enhanced / 255.0
            renorm_img = (normalized * 255).astype(np.uint8)

            # Spatial Filters
            blur_img = cv2.GaussianBlur(renorm_img, (5, 5), 0)
            med_filter = cv2.medianBlur(blur_img, 5)
            
            kernel_sharpening = np.array([[-1, -1, -1], 
                                          [-1,  9, -1], 
                                          [-1, -1, -1]])
            sharpened = cv2.filter2D(med_filter, -1, kernel_sharpening)

            # Save processed image
            cv2.imwrite(save_path, sharpened)

    print(f"Processed: {input_path} → {output_path} ✅")

# --- 3. EXECUTE PREPROCESSING ---
preprocess_folder("dataset/train", "processed/train")
preprocess_folder("dataset/valid", "processed/valid")
preprocess_folder("dataset/test", "processed/test")
print("All Preprocessing Done ✅")

# --- 4. VISUALIZATION FOR THE PROFESSOR ---
def show_results_for_professor(base_path):
    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not categories:
        print("No processed folders found to display.")
        return

    print("\n--- Visualizing Processed Results ---")
    
    # Pick a random category to showcase
    category = random.choice(categories)
    cat_path = os.path.join(base_path, category)
    images = os.listdir(cat_path)
    
    if images:
        sample_img_name = random.choice(images)
        processed_img_path = os.path.join(cat_path, sample_img_name)
        
        # Load the processed image
        processed_img = cv2.imread(processed_img_path)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.imshow(processed_rgb, cmap='gray' if len(processed_rgb.shape)==2 else None)
        plt.title(f"Final Processed Image: {category.upper()}", fontsize=15)
        plt.axis('off')
        plt.show()

# Run the showcase
show_results_for_professor("processed/train")







# Run the function
# apply_spatial_filters('your_image.jpg')

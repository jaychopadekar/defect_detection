import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset folder
dataset_path = r"C:\Users\jaych\Desktop\Steel\imgs"
augmented_images_path = r"C:\Users\jaych\Desktop\Steel\augmented_imgs"

# Create the output directory for each defect type if it doesn't exist
os.makedirs(os.path.join(augmented_images_path, 'Dent'), exist_ok=True)
os.makedirs(os.path.join(augmented_images_path, 'Crack'), exist_ok=True)
os.makedirs(os.path.join(augmented_images_path, 'No defect'), exist_ok=True)

# Define the target size for augmented images
target_size = (128, 128)  # Adjust the size as needed

# Function to preprocess images
def preprocess_and_augment_images():
    for defect_type in ['Dent', 'Crack', 'No defect']:
        defect_path = os.path.join(dataset_path, defect_type)
        images = os.listdir(defect_path)
        
        for image_name in images:
            image_path = os.path.join(defect_path, image_name)
            # Read and convert to grayscale
            img = cv2.imread(image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Save the grayscale image (optional)
            gray_img_path = os.path.join(augmented_images_path, defect_type, f'gray_{image_name}')
            cv2.imwrite(gray_img_path, gray_img)

            # Define ImageDataGenerator with augmentation options
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2],
                preprocessing_function=lambda x: x + np.random.normal(scale=0.1, size=x.shape)
            )

            # Reshape and augment
            gray_img = np.expand_dims(gray_img, axis=-1)  # Add channel dimension
            gray_img = np.expand_dims(gray_img, axis=0)   # Add batch dimension
            
            # Generate augmented images
            for i, batch in enumerate(datagen.flow(gray_img, batch_size=1)):
                augmented_image = batch[0].astype(np.uint8)

                # Resize augmented images to target size
                resized_image = cv2.resize(augmented_image, target_size)

                # Save augmented images in the respective defect type folder
                augmented_image_path = os.path.join(augmented_images_path, defect_type, f'aug_{defect_type}_{i}_{image_name}')
                cv2.imwrite(augmented_image_path, resized_image)

                # Stop after saving 5 augmented images per original image
                if i >= 5:
                    break

# Call the function
preprocess_and_augment_images()

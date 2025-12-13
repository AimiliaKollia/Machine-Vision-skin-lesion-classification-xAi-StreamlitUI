import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config

# Define the generator with your specific settings
# Note: We removed 'preprocessing_function' because our feature pipeline handles color/contrast.
# This generator focuses on GEOMETRIC variations.
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def get_augmentation_factor(class_name, class_counts, max_count):
    """
    Calculates how many augmented versions we need per image
    to reach the majority class count.
    """
    current_count = class_counts.get(class_name, 0)
    if current_count == 0: return 0

    # Example: If Max=1000 and Current=100, factor is 10.
    # We need 9 new images for every 1 original image.
    factor = int(max_count / current_count)
    return factor


def generate_augmented_images(img, count=1):
    """
    Takes an OpenCV image, converts to Keras format, generates 'count' variations,
    and returns them as a list of OpenCV images.
    """
    if count <= 0: return []

    # 1. Keras expects RGB, OpenCV is BGR. Convert for safety (though geometric ops don't care)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Keras expects 4D array (Batch Size, Height, Width, Channels)
    img_expanded = np.expand_dims(img_rgb, 0)

    augmented_images = []

    # 3. Generate
    # flow() generates batches indefinitely, so we loop 'count' times
    i = 0
    for batch in datagen.flow(img_expanded, batch_size=1):
        # Retrieve the single image from batch
        aug_img = batch[0].astype('uint8')

        # Convert back to BGR for our feature pipeline
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

        augmented_images.append(aug_img_bgr)
        i += 1
        if i >= count:
            break

    return augmented_images
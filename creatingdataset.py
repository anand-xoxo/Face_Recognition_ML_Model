drive_folder = '/content/drive/MyDrive/practice_hehe/dataset_generation/Unknown'
os.makedirs(drive_folder, exist_ok=True)
print(f"Images will be saved to: {drive_folder}")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

#Initializing ImageDataGenerator with common augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'         #filling missing pixels after transformation
)

img_path = os.path.join(drive_folder, 'cice.png')
img = image.load_img(img_path)
img_array = image.img_to_array(img)  #converting the image into an array
img_array = np.expand_dims(img_array, axis=0)  #adding a batch dimension

#Applying Augmentation
i = 0
for batch in datagen.flow(img_array, batch_size=1, save_to_dir=drive_folder, save_prefix='augmented', save_format='jpeg'):
    i += 1
    if i > 25:
        break

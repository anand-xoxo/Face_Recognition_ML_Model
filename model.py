from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#Loading the model
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/practice_hehe/face_recognition_model.h5')

#Loading and preprocessing the test image
test_img_path = '/content/drive/MyDrive/practice_hehe/test1.png'  # Replace with your test image
test_img = load_img(test_img_path, target_size=(224, 224))  # Resize to match model input size
test_img_array = img_to_array(test_img) / 255.0  # Normalize pixel values
test_img_array = np.expand_dims(test_img_array, axis=0)  # Add batch dimension

# Making a prediction
predictions = model.predict(test_img_array)
predicted_class = np.argmax(predictions)  # Get the class index with highest probability
class_labels = list(train_generator.class_indices.keys())  # Get class labels
predicted_label = class_labels[predicted_class]
confidence = np.max(predictions)
if confidence > 0.60 and class_labels[predicted_class] != 'Unknown':  # Known person
            print(f"Predicted class: {predicted_label} -- Resident")
else:  # Unknown person
            print(f"Unknown")

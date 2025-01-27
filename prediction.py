from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Path to the image you want to classify
image_path = 'C:/Users/soham/Downloads/Projects-20240722T093004Z-001/Projects/animal_classification/Animal Classification/dataset/Lion/Lion_7.jpg'

# Load the image and resize it to the target size
image = load_img(image_path, target_size=(224, 224))

# Convert the image to a numpy array and scale the pixel values
image = img_to_array(image) / 255.0
image = image.reshape((1, 224, 224, 3))  # Reshape for the model

# Load the trained model
model = tf.keras.models.load_model('C:/Users/soham/Downloads/unified mentor/model/animal_classifier.h5')

# Predict the class
prediction = model.predict(image)

# Get the class label
class_labels = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
predicted_class = class_labels[prediction.argmax()]

print(f'The predicted class is: {predicted_class}')

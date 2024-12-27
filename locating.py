import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import cv2

# Enable mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Paths to your datasets
health_path = r"C:\\Users\\laksh\\Downloads\\lungs\\TBX11K\\imgs\\health"
sick_path = r"C:\\Users\\laksh\\Downloads\\lungs\\TBX11K\\imgs\\sick"
tb_path = r"C:\\Users\\laksh\\Downloads\\lungs\\TBX11K\\imgs\\tb"

# Helper function to load data
def load_images_and_labels(folder_path, label):
    images = []
    labels = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))  # Downscale to 256x256
        img_array = tf.keras.utils.img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Load images and labels
health_images, health_labels = load_images_and_labels(health_path, 0)
sick_images, sick_labels = load_images_and_labels(sick_path, 1)
tb_images, tb_labels = load_images_and_labels(tb_path, 2)

# Combine data and labels
data = np.array(health_images + sick_images + tb_images)
labels = np.array(health_labels + sick_labels + tb_labels)

# Normalize images
data = data / 255.0

# One-hot encode labels
labels = to_categorical(labels, num_classes=3)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Handle class imbalance
class_counts = [len(health_labels), len(sick_labels), len(tb_labels)]
total_samples = sum(class_counts)
class_weights = {i: total_samples / (3 * count) for i, count in enumerate(class_counts)}

print("Class Weights:", class_weights)

# Define the MobileNetV2-based model
base_model = MobileNetV2(include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers for classification
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),  # Reduce batch size to 8
    validation_data=(X_val, y_val),
    epochs=25,
    class_weight=class_weights
)

# Save the model
model.save('tb_mobilenetv2_model.h5')

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Test model on a single image and draw bounding box
def test_model_with_box(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))  # Match input size
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classes = ['Healthy', 'Sick but Non-TB', 'Active TB']

    # Visualize the bounding box for demonstration (Dummy example)
    img_orig = cv2.imread(image_path)
    img_orig = cv2.resize(img_orig, (256, 256))
    if predicted_class == 2:  # Assuming "Active TB" is the class to highlight
        start_point = (50, 50)  # Example bounding box coordinates
        end_point = (200, 200)
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        img_orig = cv2.rectangle(img_orig, start_point, end_point, color, thickness)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {classes[predicted_class]}")
    plt.axis('off')
    plt.show()

# Test on a sample image
test_model_with_box(r"C:\\Users\\laksh\\Downloads\\lungs\\TBX11K\\imgs\\tb\\tb1126.png")
import os
import cv2
import numpy as np
from sklearn.svm import SVC

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Check if it's a directory
            if label.startswith('cat'):
                class_label = 0  # Class label for cats
            elif label.startswith('dog'):
                class_label = 1  # Class label for dogs
            else:
                continue  # Skip other files
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (100, 100))  # Resize image if necessary
                    images.append(image)
                    labels.append(class_label)
    return np.array(images), np.array(labels)

# Load training data
train_dir = "D:/prodigy tasks/task3/train"
print("Loading training data...")
X_train, y_train = load_data(train_dir)
print("Training data loaded.")

# Print the number of images for each class
print("Number of images for class 0 (cats):", np.sum(y_train == 0))
print("Number of images for class 1 (dogs):", np.sum(y_train == 1))

# Preprocess training data
# (Add any preprocessing steps here if necessary)

# Reshape the input data to have two dimensions
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Train SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_flat, y_train)
print("SVM classifier trained successfully.")


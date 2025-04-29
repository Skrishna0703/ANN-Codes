import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Configuration
DATA_DIR = '/path/to/car-object-detection-dataset'
IMG_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 32

# Load Images
def load_images(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for idx, label in enumerate(class_names):
        for img_file in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
    return np.array(images), to_categorical(labels)

X, y = load_images(DATA_DIR)
X = X / 255.0  # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
def build_model(optimizer):
    base_model = ResNet50(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(y.shape[1], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define Optimizers
optimizers = {
    "Adam": Adam(learning_rate=1e-4),
    "SGD": SGD(learning_rate=1e-4, momentum=0.9),
    "RMSprop": RMSprop(learning_rate=1e-4)
}

# Train and Evaluate
history_dict = {}
results = {}

for name, opt in optimizers.items():
    print(f"Training with {name}...")
    model = build_model(opt)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = acc
    history_dict[name] = history

# Plot Validation Accuracy
for name, history in history_dict.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy')

plt.title('Validation Accuracy by Optimizer')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Print Final Accuracies
print("\nFinal Accuracies:")
for opt_name, acc in results.items():
    print(f"{opt_name}: {acc:.4f}")

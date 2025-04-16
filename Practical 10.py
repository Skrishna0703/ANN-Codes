import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image manipulation

# Build Object Detection Model
def build_object_detection_model(input_shape, num_classes):
    """
    Builds a CNN-based object detection model. This is a *simplified*
    architecture, and in practice, you'd use a more complex one like YOLO, SSD, or Faster R-CNN.
    This example assumes a single object detection and classification task.

    Args:
        input_shape: Shape of the input image (e.g., (224, 224, 3)).
        num_classes: Number of object classes to detect.

    Returns:
        tf.keras.Model: A Keras model.
    """
    # Input layer
    input_img = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the features
    x = Flatten()(x)

    # Fully connected layers for classification and bounding box regression
    # Classification branch
    cls_branch = Dense(256, activation='relu')(x)
    cls_branch = Dropout(0.5)(cls_branch)
    cls_output = Dense(num_classes, activation='softmax', name='class_output')(cls_branch)

    # Bounding box regression branch
    bbox_branch = Dense(256, activation='relu')(x)
    bbox_branch = Dropout(0.5)(bbox_branch)
    bbox_output = Dense(4, activation='linear', name='box_output')(bbox_branch)  # 4 outputs: (x, y, width, height)

    # Define the model with two outputs: class and bounding box
    model = Model(inputs=input_img, outputs=[cls_output, bbox_output])
    return model

# Training the object detection model
def train_object_detector(model, train_data, val_data, epochs, batch_size):
    """
    Trains the object detection model.

    Args:
        model: A Keras model.
        train_data: Tuple of (images, [class_labels, bounding_boxes])
        val_data: Tuple of (images, [class_labels, bounding_boxes])
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    """
    model.compile(optimizer='adam',
                  loss={'class_output': 'categorical_crossentropy', 'box_output': 'mse'},  # Loss for both tasks
                  metrics={'class_output': 'accuracy'})

    # Prepare data for training
    train_images, [train_class_labels, train_bbox_labels] = train_data
    val_images, [val_class_labels, val_bbox_labels] = val_data

    # Train the model
    model.fit(train_images,
              {'class_output': train_class_labels, 'box_output': train_bbox_labels},
              validation_data=(val_images, {'class_output': val_class_labels, 'box_output': val_bbox_labels}),
              epochs=epochs,
              batch_size=batch_size)
    return model  # Return the trained model

# Evaluate the object detection model
def evaluate_object_detector(model, test_data):
    """
    Evaluates the object detector on a test dataset.

    Args:
        model: A trained Keras model.
        test_data: Tuple of (images, [class_labels, bounding_boxes])

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    test_images, [test_class_labels, test_bbox_labels] = test_data
    results = model.evaluate(test_images,
                           {'class_output': test_class_labels, 'box_output': test_bbox_labels},
                           verbose=0)
    return {'class_accuracy': results[1], 'box_loss': results[2]}

# Visualize the predictions
def visualize_prediction(image, predicted_class, predicted_bounding_box, class_names):
    """
    Visualizes the prediction on the image.

    Args:
        image: The original image (numpy array).
        predicted_class: The predicted class index.
        predicted_bounding_box: The predicted bounding box coordinates (x, y, width, height).
        class_names: A list of class names.
    """
    image = np.array(image * 255, dtype=np.uint8)  # Convert to uint8 for OpenCV compatibility
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    class_name = class_names[predicted_class]
    x, y, w, h = predicted_bounding_box
    x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
    cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Predict the object class and bounding box
def predict_object_detection(model, image):
    """
    Predicts the class and bounding box for a given image.

    Args:
        model: The trained object detection model.
        image: The input image to make predictions on.

    Returns:
        predicted_class: The predicted class index.
        predicted_bounding_box: The predicted bounding box coordinates (x, y, width, height).
    """
    # Preprocess the input image (resize, normalize, etc.)
    image_resized = cv2.resize(image, (224, 224))  # Assuming the model expects 224x224 input size
    image_normalized = image_resized / 255.0  # Normalize the image to [0, 1]
    image_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

    # Make predictions using the trained model
    class_predictions, bbox_predictions = model.predict(image_batch)

    # Get the class with the highest probability
    predicted_class = np.argmax(class_predictions, axis=-1)[0]  # Get the index of the max value
    predicted_bounding_box = bbox_predictions[0]  # The model predicts the bounding box

    return predicted_class, predicted_bounding_box

# Main function to run the training and testing process
def main():
    input_shape = (224, 224, 3)  # Example image size
    num_classes = 10  # Example: 10 different object classes
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

    # Build the model
    model = build_object_detection_model(input_shape, num_classes)
    model.summary()

    # Create dummy data for demonstration
    num_train_samples = 100
    num_val_samples = 20
    num_test_samples = 30
    train_images = np.random.rand(num_train_samples, 224, 224, 3)
    train_class_labels = np.random.randint(0, num_classes, size=(num_train_samples, num_classes))  # One-hot encoded
    train_bbox_labels = np.random.rand(num_train_samples, 4)  # (x, y, width, height)
    val_images = np.random.rand(num_val_samples, 224, 224, 3)
    val_class_labels = np.random.randint(0, num_classes, size=(num_val_samples, num_classes))
    val_bbox_labels = np.random.rand(num_val_samples, 4)
    test_images = np.random.rand(num_test_samples, 224, 224, 3)
    test_class_labels = np.random.randint(0, num_classes, size=(num_test_samples, num_classes))
    test_bbox_labels = np.random.rand(num_test_samples, 4)

    train_data = (train_images, [train_class_labels, train_bbox_labels])
    val_data = (val_images, [val_class_labels, val_bbox_labels])
    test_data = (test_images, [test_class_labels, test_bbox_labels])

    # Train the model
    trained_model = train_object_detector(model, train_data, val_data, epochs=10, batch_size=32)

    # Make a prediction
    sample_image = np.random.rand(224, 224, 3)
    predicted_class, predicted_bounding_box = predict_object_detection(trained_model, sample_image)
    print(f"Predicted Class: {predicted_class}, Predicted Bounding Box: {predicted_bounding_box}")

    # Evaluate the model
    evaluation_metrics = evaluate_object_detector(trained_model, test_data)
    print("Evaluation Metrics:", evaluation_metrics)

    # Visualize the prediction
    visualize_prediction(sample_image, predicted_class, predicted_bounding_box, class_names)

if __name__ == '__main__':
    main()

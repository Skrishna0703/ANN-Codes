import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cnn_model.keras")

# Preprocess and predict
def preprocess_and_predict(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# GUI App
class CNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Image Classifier (MNIST)")

        self.label = tk.Label(root, text="Upload a Digit Image (28x28)", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img_resized = img.resize((100, 100))
            img_tk = ImageTk.PhotoImage(img_resized)

            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            predicted_class = preprocess_and_predict(file_path)
            self.result_label.config(text=f"Predicted Digit: {predicted_class}")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = CNNApp(root)
    root.mainloop()

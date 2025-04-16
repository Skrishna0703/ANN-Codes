import numpy as np
import tkinter as tk

# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Training the XOR model
def train_model():
    global wh, bh, wo, bo
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    wh = np.random.uniform(size=(2, 2))
    bh = np.random.uniform(size=(1, 2))
    wo = np.random.uniform(size=(2, 1))
    bo = np.random.uniform(size=(1, 1))

    for _ in range(10000):
        h_input = np.dot(X, wh) + bh
        h_output = sigmoid(h_input)

        o_input = np.dot(h_output, wo) + bo
        output = sigmoid(o_input)

        error = y - output
        d_output = error * sigmoid_deriv(output)

        error_hidden = d_output.dot(wo.T)
        d_hidden = error_hidden * sigmoid_deriv(h_output)

        wo += h_output.T.dot(d_output) * 0.1
        bo += np.sum(d_output, axis=0, keepdims=True) * 0.1
        wh += X.T.dot(d_hidden) * 0.1
        bh += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

# Predict XOR output for input
def predict():
    i1 = int(entry1.get())
    i2 = int(entry2.get())
    x = np.array([[i1, i2]])
    h = sigmoid(np.dot(x, wh) + bh)
    o = sigmoid(np.dot(h, wo) + bo)
    output_label.config(text="Output: " + str(round(o[0][0])))

# GUI setup
root = tk.Tk()
root.title("XOR BPN")
root.geometry("250x200")

tk.Label(root, text="Input 1 (0/1):").pack()
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="Input 2 (0/1):").pack()
entry2 = tk.Entry(root)
entry2.pack()

tk.Button(root, text="Predict XOR", command=predict).pack(pady=10)

output_label = tk.Label(root, text="Output: ")
output_label.pack()

train_model()  # Train once at start
root.mainloop()

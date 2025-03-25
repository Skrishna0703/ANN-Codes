
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Generate input values
x = np.linspace(-5, 5, 400)

# Compute activation function values
y_functions = {
    "Linear": linear(x),
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "ReLU": relu(x)
}

# Colors for plots
colors = ["blue", "red", "green", "purple"]

# Create subplots
plt.figure(figsize=(10, 6))

for i, (name, y) in enumerate(y_functions.items(), start=1):
    plt.subplot(2, 2, i)  # Arrange in 2 rows, 2 columns
    plt.plot(x, y, label=name, color=colors[i-1], linewidth=2)
    plt.title(name, fontsize=12, fontweight="bold")
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=0.7)
    plt.axvline(x=0, color='black', linestyle='dashed', linewidth=0.7)
    plt.grid(alpha=0.4)
    plt.legend()

plt.suptitle("Comparison of Activation Functions", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout
plt.show()

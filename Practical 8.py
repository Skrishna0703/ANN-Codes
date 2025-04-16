import numpy as np

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR Inputs and Targets
X = np.array([[0,0], [0,1], [1,0], [1,1]])   # Inputs
T = np.array([[0], [1], [1], [0]])           # Targets

# Initialize weights and biases (Step 1)
np.random.seed(42)
wh = np.random.uniform(-1, 1, (2, 2))   # Input to Hidden
bh = np.random.uniform(-1, 1, (1, 2))   # Bias of hidden
wo = np.random.uniform(-1, 1, (2, 1))   # Hidden to Output
bo = np.random.uniform(-1, 1, (1, 1))   # Bias of output

alpha = 0.1  # Learning rate
epochs = 10000

# Training loop (Step 2 to 10)
for epoch in range(epochs):
    # Step 4-5: Feedforward
    hin = np.dot(X, wh) + bh
    hout = sigmoid(hin)

    oin = np.dot(hout, wo) + bo
    y = sigmoid(oin)

    # Step 6: Output layer error
    error_output = (T - y) * sigmoid_derivative(y)   # δk

    # Step 7: Hidden layer error
    error_hidden = error_output.dot(wo.T) * sigmoid_derivative(hout)  # δj

    # Step 8: Update weights and biases
    wo += hout.T.dot(error_output) * alpha
    bo += np.sum(error_output, axis=0, keepdims=True) * alpha

    wh += X.T.dot(error_hidden) * alpha
    bh += np.sum(error_hidden, axis=0, keepdims=True) * alpha

# Final Predictions
print("Final predictions after training:")
for i in range(4):
    test_input = X[i]
    hidden = sigmoid(np.dot(test_input, wh) + bh)
    output = sigmoid(np.dot(hidden, wo) + bo)
    print(f"Input: {test_input}, Predicted Output: {np.round(output[0])}, Actual: {T[i][0]}")

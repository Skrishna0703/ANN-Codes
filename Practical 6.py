import numpy as np 

# Sigmoid activation function and its derivative 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 

def sigmoid_derivative(x): 
    return x * (1 - x) 

# Sample dataset: XOR problem 
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]]) 

y = np.array([[0], 
              [1], 
              [1], 
              [0]]) 

# Seed for reproducibility 
np.random.seed(1) 

# Network architecture 
input_neurons = 2 
hidden_neurons = 3 
output_neurons = 1 

# Weight and bias initialization 
W1 = np.random.uniform(size=(input_neurons, hidden_neurons)) 
b1 = np.random.uniform(size=(1, hidden_neurons)) 

W2 = np.random.uniform(size=(hidden_neurons, output_neurons)) 
b2 = np.random.uniform(size=(1, output_neurons)) 

# Training parameters 
epochs = 10000 
lr = 0.1 

# Training loop 
for epoch in range(epochs): 
    # Forward Propagation 
    z1 = np.dot(X, W1) + b1 
    a1 = sigmoid(z1) 
    
    z2 = np.dot(a1, W2) + b2 
    a2 = sigmoid(z2) 
    
    # Loss computation 
    loss = y - a2 
    
    # Backpropagation 
    d_a2 = loss * sigmoid_derivative(a2) 
    dW2 = np.dot(a1.T, d_a2) 
    db2 = np.sum(d_a2, axis=0, keepdims=True) 
    
    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1) 
    dW1 = np.dot(X.T, d_a1) 
    db1 = np.sum(d_a1, axis=0, keepdims=True) 
    
    # Update weights and biases 
    W2 += lr * dW2 
    b2 += lr * db2 
    W1 += lr * dW1 
    b1 += lr * db1 
    
    # Print the error every 1000 epochs
    if epoch % 1000 == 0: 
        error = np.mean(np.square(loss)) 
        print(f"Epoch {epoch}, Loss: {error:.4f}") 

# Final output 
print("\nFinal Predictions:") 
print(np.round(a2))

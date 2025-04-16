import numpy as np

def bipolar_to_binary(vector):
    return np.where(vector == -1, 0, 1)

def binary_to_bipolar(vector):
    return np.where(vector == 0, -1, 1)

def train_bam(X, Y):
    """ Train BAM by computing the weight matrix """
    W = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(X.shape[0]):
        W += np.outer(X[i], Y[i])
    return W

def recall_bam(X, W, mode="forward"):
    """ Recall associated patterns """
    if mode == "forward":
        return np.sign(X @ W)
    elif mode == "backward":
        return np.sign(X @ W.T)
    else:
        raise ValueError("Invalid mode. Use 'forward' or 'backward'.")

# Define new bipolar input-output pairs
X = np.array([[1, -1, 1], [-1, 1, -1]])  # New Input patterns
Y = np.array([[-1, 1, -1], [1, -1, 1]])  # New Associated output patterns

# Train BAM
W = train_bam(X, Y)

# Recall from input to output (Forward direction)
Y_recalled = recall_bam(X, W, mode="forward")
print("Recalled Output (Forward):\n", Y_recalled)

# Recall from output to input (Backward direction)
X_recalled = recall_bam(Y, W, mode="backward")
print("Recalled Input (Backward):\n", X_recalled)

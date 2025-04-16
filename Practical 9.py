import numpy as np
def train_hopfield_network(patterns):  
    # Number of patterns and number of neurons
    num_patterns = len(patterns)
    num_neurons = patterns[0].shape[0]
    # Initialize the weight matrix
    W = np.zeros((num_neurons, num_neurons))
    # Iterate over all pattern pairs
    for p in range(num_patterns):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i != j:
                    # Bipolar Hebbian learning rule
                    W[i, j] += patterns[p][i] * patterns[p][j]
    return W
def recall_hopfield_network(W, initial_state, max_iterations=100):   
    num_neurons = W.shape[0]
    state = initial_state.copy()  # Create a copy to avoid modifying the original
    for iteration in range(max_iterations):
        # Asynchronous update: update neurons one at a time in random order
        neuron_order = np.random.permutation(num_neurons)
        state_changed = False # Flag to check if any state changed in this iteration
        for i in neuron_order:
            # Calculate the net input to neuron i
            net_input = np.dot(W[i, :], state) # No need to add state[i]*W[i,i] since W[i,i] is 0
            # Update the neuron's state using the activation function (sign function)
            new_state_i = 1 if net_input > 0 else -1
            if new_state_i != state[i]:
                state[i] = new_state_i
                state_changed = True # Set the flag to True if state changed
        if not state_changed:
            break # If no state changed, the network has converged
    return state
def test_hopfield_network():
    # Define the 4 patterns to be stored
    pattern1 = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    pattern2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    pattern3 = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    pattern4 = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    patterns = [pattern1, pattern2, pattern3, pattern4]
    # Train the Hopfield network
    W = train_hopfield_network(patterns)
    print("Weight Matrix W:\n", W)
    # Test the network with a noisy version of pattern1
    noisy_pattern1 = np.array([1, -1, -1, -1, 1, -1, 1, -1]) # Introduce one error.
    print("\nOriginal Pattern 1:", pattern1)
    print("Noisy Pattern 1:  ", noisy_pattern1)
    # Recall the stored pattern from the noisy input
    recalled_pattern = recall_hopfield_network(W, noisy_pattern1)
    print("Recalled Pattern 1:", recalled_pattern)
    # Check if the network successfully recalled the original pattern
    if np.array_equal(recalled_pattern, pattern1):
        print("Test Passed: Network successfully recalled Pattern 1")
    else:
        print("Test Failed: Network failed to recall Pattern 1")
    # Test with a pattern that is not very close to any of the stored patterns
    test_pattern = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    print("\nTest Pattern:", test_pattern)
    recalled_pattern = recall_hopfield_network(W, test_pattern)
    print("Recalled Pattern:", recalled_pattern)
    print("This pattern should converge to one of the stored patterns,")
    print("although it might take more iterations.  It is not guaranteed to converge")
    print("to the closest pattern.")
    print("Testing with max_iterations=10:")
    recalled_pattern_short = recall_hopfield_network(W, test_pattern, max_iterations=10)
    print("Recalled Pattern (10 iterations):", recalled_pattern_short)
if __name__ == "__main__":
    test_hopfield_network()

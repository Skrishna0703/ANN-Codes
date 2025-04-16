import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# Generate a simple dataset (AND logic gate)
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Input features
y = np.array([0, 0, 0, 1])  # Labels (AND logic gate)

# Train a perceptron model
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X, y)

# Plot decision regions
plt.figure(figsize=(6, 4))
plot_decision_regions(X, y, clf=model, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')
plt.show()

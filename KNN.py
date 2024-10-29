from collections import Counter
import numpy as np
import pandas as pd
class CustomKNN:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class labels for the input data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict the class label for a single input instance."""
        # Calculate distances from x to all points in the training set
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Convert labels to a hashable type if they are not
        # Make sure labels are either strings or integers
        k_nearest_labels = [label.item() if isinstance(label, np.ndarray) else label for label in k_nearest_labels]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
# Example usage
if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('./dataset.csv')
    # Prepare features (X) and labels (y)
    X = data.iloc[:, :-1].values  # All rows, all columns except the last one
    y = data.iloc[:, -1].values   # All rows, only the last column (gesture labels)
    # Create an instance of CustomKNN
    knn = CustomKNN(k=3) 
    # Fit the model
    knn.fit(X, y)
    sample_data = np.array([[338.27794587544304, 356.3608903714182, 326.48402123972147, 317.0225736078043, 300.0044154933058]])  # Replace with actual angles
    prediction = knn.predict(sample_data)
    print(f'Predicted Gesture: {prediction[0]}')

import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        model_prediction = self.get_model_prediction(X, initial_weights)
        grads = np.zeros_like(initial_weights)

        for i in range(len(initial_weights)):
            grads[i] = self.get_derivative(
                model_prediction,
                Y,
                len(X),
                X,
                i,
            )
        
        updated_weights = initial_weights.copy()
        k = 0
        while k < num_iterations:
            updated_weights -= self.learning_rate * grads
            model_prediction = self.get_model_prediction(X, updated_weights)
            for i in range(len(initial_weights)):
                grads[i] = self.get_derivative(
                    model_prediction,
                    Y,
                    len(X),
                    X,
                    i,
                )
            k = k + 1
        
        return np.round(updated_weights, 5)

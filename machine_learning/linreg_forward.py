import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:

        preds = X @ weights

        return np.round(preds, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:

        error = np.mean(np.square(model_prediction - ground_truth))

        return np.round(error, 5)

import time
import numpy as np

class LogRes:

    def __init__(self, feature_count=19):
        np.random.seed(int(time.time()))
        self.W = np.zeros(shape=feature_count)

        self.b = 0

    def prepare_data(self, x_train, y_train):
        x_train = np.array(x_train, 'float64')
        y_train = np.array(y_train, 'float64')
        return x_train, y_train

    def train(self, x_train, y_train, n_epochs=1000000, lr=0.001):
        n_samples = x_train.shape[0]

        for i in range(n_epochs):
            scores = x_train.dot(self.W) + self.b
            sig_scores = 1 / (1 + np.exp(-scores))

            dw = (1 / n_samples) * np.dot(x_train.T, (sig_scores - y_train))
            db = (1 / n_samples) * np.sum(sig_scores - y_train)

            self.W -= dw * lr
            self.b -= db * lr

    def test(self, x_train):
        # e produce our estimated scores
        scores = x_train.dot(self.W) + self.b

        # We use the sigmoid function to fit the scores to values between 1 and 0, to convert them into probabilities.

        sig_scores = 1 / (1 + np.exp(-scores))

        # To make our predictions ready, we arrange the scores less than 0.5 as 0 and the remaining scores as 1.
        predictions = np.zeros(shape=sig_scores.shape)
        predictions[sig_scores >= 0.5] = 1
        predictions[sig_scores < 0.5] = 0

        return predictions

    def test_detailed(self, x_train):
        # we produce our estimated scores
        scores = x_train.dot(self.W) + self.b

        sig_scores = 1 / (1 + np.exp(-scores))

        predictions = np.zeros(shape=sig_scores.shape)
        predictions[sig_scores >= 0.5] = 1
        predictions[sig_scores < 0.5] = 0

        return predictions, sig_scores, scores

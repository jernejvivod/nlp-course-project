import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from feature_engineering import get_markov_model, get_conditional_probabilities

class ClfWrap(BaseEstimator, ClassifierMixin):
    """
    Wrap classifier to include Markov model and 
    conditional probabilities in predictions.
    
    Args:
        clf (obj): TODO
        n_look_back (int): TODO
        alpha (float): TODO
        beta (float): TODO
        thresh (float): TODO
        predict_proba (bool): TODO

    Attributes:
        clf (obj): TODO
        n_look_back (int): TODO
        alpha (float): TODO
        beta (float): TODO
        thresh (float): TODO
        predict_proba (bool): TODO
    """

    def __init__(self, clf, n_look_back=4, alpha=0.0, beta=0.0, thresh=0.5, predict_proba=False):
        self.clf = clf
        self.n_look_back = n_look_back
        self.alpha = alpha
        self.beta = beta
        self.thresh = thresh
        self.predict_proba = predict_proba


    def fit(self, X, y):
        """
        Fit classifier to training data.

        Args:
            X (numpy.ndarray): Training data samples
            y (numpy.ndarray): Training data labels

        Returns:
            (obj): Reference to self
        """

        # Get number of unique classes.
        self.num_classes = len(np.unique(y))

        # Get Markov model and conditional probabilities.
        self.markov_mod = get_markov_model(y)
        self.cond_prob = get_conditional_probabilities(y, self.n_look_back)

        # Fit wrapped classifier.
        self.clf = self.clf.fit(X, y)

        # Return reference to self.
        return self


    def predict(self, X, y=None):
        """
        Predict labels of new data.

        Args:
            X (numpy.ndarray): Data for which to predict classes

        Returns:
            (numpy.ndarray): Predicted labels
        """

        # Initialize predictions array.
        if self.predict_proba:
            predictions = np.empty((X.shape[0], self.num_classes), dtype=float)
        else:
            predictions = np.empty(X.shape[0], dtype=int)
        
        # Go over samples.
        for idx, sample in enumerate(X):
            
            # Predict probability for next sample.
            pred_clf = self.clf.predict_proba(sample.reshape(1, -1))[0]
            
            if idx > 0:
                # If not first sample, use make prediction using Markov model.
                if self.predict_proba:
                    pred_markov_mod = self.markov_mod[np.argmax(predictions[idx-1])]
                else:
                    pred_markov_mod = self.markov_mod[predictions[idx-1]]

                if idx >= self.n_look_back:
                    # If sample index large enough to use conditional probabilities,
                    # use computed conditional probabilities to make prediction.
                    if self.predict_proba:
                        patt = predictions[idx-self.n_look_back:idx]
                        pred_cond_prob = self.cond_prob[str(np.argmax(patt, axis=1))]
                    else:
                        patt = predictions[idx-self.n_look_back:idx]
                        pred_cond_prob = self.cond_prob[str(patt)]

                    # Combine predictions of classifier, Markov model and conditional probabilities.
                    pred_comb = (1-self.alpha-self.beta)*pred_clf + self.alpha*pred_markov_mod + self.beta*pred_cond_prob
                else:
                    pred_comb = (1-self.alpha)*pred_clf + self.alpha*pred_markov_mod
            else:
                pred_comb = pred_clf
            
            if self.predict_proba:
                predictions[idx, :] = pred_comb
            else:
                predictions[idx] = int(pred_comb[1] > self.thresh)

        return predictions
    
    def score(self, X, y, sample_weight=None):
        """
        Score predictions on training data.
        
        Args:
            X (numpy.ndarray): Test data
            y (numpy.ndarray): True labels
            sample_weight (numpy.ndarray): Sample weights for scoring predictions

        Returns:
            (float): Accuracy scores of predictions
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)



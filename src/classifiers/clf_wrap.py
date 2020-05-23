import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from feature_engineering import get_markov_model, get_conditional_probabilities

class ClfWrap(BaseEstimator, ClassifierMixin):
    """
    Wrap classifier to include Markov model and 
    conditional probabilities in predictions.
    
    Args:
        clf (obj): Classifier to wrap
        n_look_back (int): How many predictions to look back when computing
        conditional probabilities.
        alpha (float): Weight assigned to Markov model prediction.
        beta (float): Weight assigned to conditional probabilities' prediction.

    Attributes:
        clf (obj): Classifier to wrap
        n_look_back (int): How many predictions to look back when computing
        conditional probabilities.
        alpha (float): Weight assigned to Markov model prediction.
        beta (float): Weight assigned to conditional probabilities' prediction.
    """
    
    def __init__(self, clf, n_look_back=4, alpha=0.06, beta=0.07):
    # def __init__(self, clf, n_look_back=4, alpha=0.0, beta=0.0):
        self.clf = clf
        self.n_look_back = n_look_back
        self.alpha = alpha
        self.beta = beta


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

        # Initialize array for predictions.
        predictions = np.empty(X.shape[0], dtype=int)
        
        # Go over samples.
        for idx, sample in enumerate(X):

            # Predict probability for next sample using classification algorithm.
            pred_clf = self.clf.predict_proba(sample.reshape(1, -1))[0]
            
            if idx > 0:
                # If not first sample, use make prediction using Markov model.
                pred_markov_mod = self.markov_mod[predictions[idx-1]]

                if idx >= self.n_look_back:
                    # If sample index large enough to use conditional probabilities,
                    # use computed conditional probabilities to make prediction.
                    patt = predictions[idx-self.n_look_back:idx]
                    
                    # Check if prediction in dictionary.
                    if str(patt) in self.cond_prob:
                        pred_cond_prob = self.cond_prob[str(patt)]

                        # Combine predictions of classifier, Markov model and conditional probabilities.
                        pred_comb = (1-self.alpha-self.beta)*pred_clf + self.alpha*pred_markov_mod + self.beta*pred_cond_prob
                        predictions[idx] = np.argmax(pred_comb)
                    else:
                        pred_comb = (1-self.alpha)*pred_clf + self.alpha*pred_markov_mod
                        predictions[idx] = np.argmax(pred_comb)
                else:
                    pred_comb = (1-self.alpha)*pred_clf + self.alpha*pred_markov_mod
                    predictions[idx] = np.argmax(pred_comb)
            else:
                # Set prediction as most probable class.
                predictions[idx] = np.argmax(pred_clf)
        
        # Return array of predictions.
        return predictions
    
    
    def predict_proba(self, X, y=None):
        """
        Predict probabilities of labels of new data.

        Args:
            X (numpy.ndarray): Data for which to predict class probabilities.

        Returns:
            (numpy.ndarray): Predicted labels
        """

        # Initialize predictions array.
        predictions = np.empty((X.shape[0], self.num_classes), dtype=float)
        
        # Go over samples.
        for idx, sample in enumerate(X):
            
            # Predict probability for next sample.
            pred_clf = self.clf.predict_proba(sample.reshape(1, -1))[0]
            
            if idx > 0:
                # If not first sample, use make prediction using Markov model.
                pred_markov_mod = self.markov_mod[np.argmax(predictions[idx-1])]

                if idx >= self.n_look_back:
                    # If sample index large enough to use conditional probabilities,
                    # use computed conditional probabilities to make prediction.
                    patt = predictions[idx-self.n_look_back:idx]
                    pred_cond_prob = self.cond_prob[str(np.argmax(patt, axis=1))]

                    # Combine predictions of classifier, Markov model and conditional probabilities.
                    pred_comb = (1-self.alpha-self.beta)*pred_clf + self.alpha*pred_markov_mod + self.beta*pred_cond_prob
                else:
                    pred_comb = (1-self.alpha)*pred_clf + self.alpha*pred_markov_mod
            else:
                pred_comb = pred_clf
            
            # Add computed probabilities to array of predictions.
            predictions[idx, :] = pred_comb
        
        # Return array of predictions.
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


    def score_features(self, f_to_name):
        """
        Score feature importances.

        Args:
            f_to_name (dict): Dictionary mapping feature enumerations such as 'f0', 'f1', ...
            to feature names.

        Returns:
            (dict): Dictionary mapping feature names as defined in f_to_name parameter to
            their estimated importances.
        """
        
        if self.name in {'gboosting', 'logreg'}:
            if self.name == 'gboosting':
                return self.clf.score_features(f_to_name)
            elif self.name == 'logreg':
                weights = np.abs(self.clf.coef_[0])/sum(np.abs(self.clf.coef_[0]))
                return {f_to_name['f' + str(idx)] : weights[idx] for idx in range(len(weights))} 
        else:
            raise(NotImplementedError('Feature scoring not implemented for this type of classifier'))


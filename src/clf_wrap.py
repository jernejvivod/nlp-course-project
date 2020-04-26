import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from feature_engineering import get_markov_model, get_conditional_probabilities

class ClfWrap(BaseEstimator, ClassifierMixin):

    def __init__(self, clf, n_look_back=4, alpha=0.0, beta=0.0, thresh=0.5, predict_proba=False):
        self.clf = clf
        self.n_look_back = n_look_back
        self.alpha = alpha
        self.beta = beta
        self.thresh = thresh
        self.predict_proba = predict_proba

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.markov_mod = get_markov_model(y)
        self.cond_prob = get_conditional_probabilities(y, self.n_look_back)
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        
        if self.predict_proba:
            predictions = np.empty((X.shape[0], self.num_classes), dtype=float)
        else:
            predictions = np.empty(X.shape[0], dtype=int)

        for idx, sample in enumerate(X):
            
            pred_clf = self.clf.predict_proba(sample.reshape(1, -1))[0]
            
            if idx > 0:
                if self.predict_proba:
                    pred_markov_mod = self.markov_mod[np.argmax(predictions[idx-1])]
                else:
                    pred_markov_mod = self.markov_mod[predictions[idx-1]]
                if idx >= self.n_look_back:
                    if self.predict_proba:
                        patt = predictions[idx-self.n_look_back:idx]
                        pred_cond_prob = self.cond_prob[str(np.argmax(patt, axis=1))]
                    else:
                        patt = predictions[idx-self.n_look_back:idx]
                        pred_cond_prob = self.cond_prob[str(patt)]

                    pred_comb = (1-self.alpha-self.beta)*pred_clf + self.alpha*pred_markov_mod + self.beta*pred_cond_prob
                else:
                    pred_comb = (1-self.alpha)*pred_clf + self.alpha*pred_markov_mod
            else:
                pred_comb = pred_clf
            
            if self.predict_proba:
                predictions[idx, :] = pred_comb
            else:
                # predictions[idx] = np.argmax(pred_comb)
                predictions[idx] = int(pred_comb[1] > self.thresh)


            

        return predictions

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)



import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

data = np.load('../data/cached/data_book_relevance.npy')
target = np.load('../data/cached/target_book_relevance.npy')


X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')

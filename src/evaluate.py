import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from clf_wrap import ClfWrap


def save_results(kind, accs=None, clf_reports=None, acc_names=None, clf_reports_names=None):
    """
    Store specified results to file.

    Args:
        kind (str): Evaluation method used to produce the results (cross-validation, train-test split, ...)
        accs (list): List of accuracies for evaluated models
        clf_reports (list): List of classification reports to write to file.
        acc_names (list): List of names of models that produced the specified accuracies
        clf_reports_names (list): List of names of models that produced the specified classification reports
    """

    # Write results to file.
    with open('../results/results.txt', 'a') as f:
        f.write('##########\n')
        f.write('Date: {0}\n'.format(datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
        f.write('##########\n\n')
        f.write('Evaluation method: {0}\n'.format(kind))
        if accs:
            for idx, acc in enumerate(accs):
                if acc_names:
                    f.write(acc_names[idx] + ': ')
                f.write(str(acc) + '\n')
        if clf_reports:
            for idx, clf_report in enumerate(clf_reports):
                if clf_reports_names:
                    f.write(clf_reports_names[idx] + ': ')
                f.write(clf_report)
        f.write('\n')


def evaluate(data, target, clf, eval_method):
    """
    Evaluate model on dataset using either cross-validation or train-test split.
    Write results to file in form of classification accuracy as well as a
    classification report.

    Args:
        data (numpy.ndarray): Dataset to use (extracted features)
        target (numpy.ndarray): Labels for the samples
        clf (obj): Classification model to evaluate
        eval_method (str): Evaluation method to use. Valid values are 'tts' for train-test
        split and 'cv' for cross-validation
    """

    # Initialize pipeline template.
    clf_pipeline = Pipeline([('scaling', StandardScaler())])
    
    # Initialize classifier.
    clf_wrapped = ClfWrap(clf)
    clf_wrapped.name = clf.name
    
    # Add classifier to pipeline.
    clf_eval = clone(clf_pipeline)
    clf_eval.steps.append(['smt', SMOTETomek()])
    clf_eval.steps.append(['clf', clf_wrapped])

    # Initialize baseline classifiers.
    clf_baseline_majority = clone(clf_pipeline)
    clf_baseline_majority.steps.append(['clf', DummyClassifier(strategy='most_frequent')])
    clf_baseline_strat = clone(clf_pipeline)
    clf_baseline_strat.steps.append(['clf', DummyClassifier(strategy='stratified')])
    clf_baseline_prior = clone(clf_pipeline)
    clf_baseline_prior.steps.append(['clf', DummyClassifier(strategy='prior')])
    clf_baseline_uniform = clone(clf_pipeline)
    clf_baseline_uniform.steps.append(['clf', DummyClassifier(strategy='uniform')])

    if eval_method == 'tts':
        # If performing train-test split.

        # Set path to discussions for retrieving fp, fn, tp and tn messages.
        discussions_path = '../data/discussions.xlsx'

        # Get data indices for retrieving falsely classified messages.
        data_ind = np.arange(data.shape[0])

        # Get training and test data.
        data_train, data_test, target_train, target_test, _, idxs_test = train_test_split(data, target, data_ind, shuffle=False, test_size=0.1)

        # Evaluate classifier.
        res_pred = clf_eval.fit(data_train, target_train).predict(data_test)
        res_eval = metrics.accuracy_score(res_pred, target_test)
        
        # Get indices of messages representing fp, fn, tp and tn.
        idx_fail_fp = idxs_test[np.logical_and(res_pred == 1, target_test == 0)]
        idx_fail_fn = idxs_test[np.logical_and(res_pred == 0, target_test == 1)]
        idx_succ_tp = idxs_test[np.logical_and(res_pred == 0, target_test == 0)]
        idx_succ_tn = idxs_test[np.logical_and(res_pred == 1, target_test == 1)]
        
        # Save fp, fn, tp and tn messages to results folder as .xlsx files.
        sheet_raw = pd.read_excel(discussions_path)
        fp = sheet_raw.loc[idx_fail_fp, :].dropna(axis='columns').to_excel('../results/fp.xlsx')
        fn = sheet_raw.loc[idx_fail_fn, :].dropna(axis='columns').to_excel('../results/fn.xlsx') 
        tp = sheet_raw.loc[idx_succ_tp, :].dropna(axis='columns').to_excel('../results/tp.xlsx') 
        tn = sheet_raw.loc[idx_succ_tn, :].dropna(axis='columns').to_excel('../results/tn.xlsx') 

        # Evaluate baseline classifiers.
        res_baseline_majority = clf_baseline_majority.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_strat = clf_baseline_strat.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_prior = clf_baseline_prior.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_uniform = clf_baseline_uniform.fit(data_train, target_train).score(data_test, target_test)
        
        # Produce classification report.
        clf_report_eval = metrics.classification_report(target_test, res_pred, target_names=['Yes', 'No'])
        clf_report_baseline = metrics.classification_report(target_test, clf_baseline_uniform.predict(data_test), target_names=['Yes', 'No'])
        
        # Save results to file. 
        # Save accuracies for evaluated model, uniform baseline model and majority baseline model.
        # Save classification reports for evaluated model and uniform baseline model.
        save_results(kind='tts', accs=[res_eval, res_baseline_uniform, res_baseline_majority], 
                     clf_reports=[clf_report_eval, clf_report_baseline], 
                     acc_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'],
                     clf_reports_names=[clf_eval['clf'].name, 'Uniform classifier'])
        

    elif eval_method == 'cv':
        # If performing cross-validation.

        # Set number of splits
        N_SPLITS = 5

        # Initialize score accumulators.
        score_cv_eval = 0
        score_cv_baseline_majority = 0
        score_cv_baseline_strat = 0
        score_cv_baseline_prior = 0
        score_cv_baseline_uniform = 0

        # Initialize fold index.
        idx = 0
        for train_idx, test_idx in KFold(n_splits=N_SPLITS, shuffle=False).split(data, target):

            # Evaluate classifier.
            score_cv_eval += clf_eval.fit(data[train_idx, :], target[train_idx]).score(data[test_idx, :], target[test_idx])

            # Evaluate baseline classifiers.
            score_cv_baseline_majority += clf_baseline_majority.fit(data[train_idx, :], target[train_idx]).score(data[test_idx, :], target[test_idx])
            score_cv_baseline_strat += clf_baseline_strat.fit(data[train_idx, :], target[train_idx]).score(data[test_idx, :], target[test_idx])
            score_cv_baseline_prior += clf_baseline_prior.fit(data[train_idx, :], target[train_idx]).score(data[test_idx, :], target[test_idx])
            score_cv_baseline_uniform += clf_baseline_uniform.fit(data[train_idx, :], target[train_idx]).score(data[test_idx, :], target[test_idx])

            # Increment fold index and print progress.
            idx += 1
            print("done {0}/{1}".format(idx, N_SPLITS))

        # Normalize scores.
        res_eval = score_cv_eval / N_SPLITS
        res_baseline_majority = score_cv_baseline_majority / N_SPLITS
        res_baseline_strat = score_cv_baseline_strat / N_SPLITS
        res_baseline_prior = score_cv_baseline_prior / N_SPLITS
        res_baseline_uniform = score_cv_baseline_uniform / N_SPLITS

        # Save results to file.
        save_results(kind='cv', accs=[res_eval, res_baseline_uniform, res_baseline_majority], 
                     acc_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'])


def plot_roc(data, target, clf):
    """
    Plot ROC curve using train-test split and save results to file.
    
    Args:
        data (numpy.ndarray): Dataset to use (extracted features)
        target (numpy.ndarray): Labels for the samples
        clf (obj): Classification model to evaluate
    """
    
    # Initialize classifier and pipeline.
    clf_wrapped = ClfWrap(clf, predict_proba=True)
    clf_eval = Pipeline([('scaling', StandardScaler()), ('clf', clf_wrapped)])

    # Get training and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, shuffle=False, test_size=0.1)

    # Evaluate classifier to get probabilities.
    scores = clf_eval.fit(data_train, target_train).predict(data_test)
    
    # Get false positive rates, true positive rates and thresholds.
    fpr, tpr, thresholds = metrics.roc_curve(target_test, scores[:, 1], pos_label=1)

    # Compute AUC.
    roc_auc = metrics.roc_auc_score(target_test, scores[:, 1])
    
    # Plot ROC curve. 
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {0:4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('../results/plots/roc.png')
    plt.clf()
    plt.close()


def confusion_matrix(data, target, clf, class_names, title, cm_save_path):
    """
    Plot and save confuction matrix for specified classifier.

    Args:
        data (numpy.ndarray): Data samples
        target (numpy.ndarray): Data labels (target variable values)
        clf (object): Classifier for which to plot the confuction matrix.
        class_names (list): List of class names
        title (str): Plot title
        cm_save_path (str): Path for saving the confusion matrix plot
    """

    # Initialize random forest classifier, apply wrapper and add to pipeline.
    clf_wrapped = ClfWrap(clf)
    clf_eval = Pipeline([('scaling', StandardScaler()), ('clf', clf_wrapped)])

    # Split data into training and test sets.
    data_train, data_test, target_train, target_test = train_test_split(data, target, shuffle=False)

    # Fit model.
    clf_eval.fit(data_train, target_train)
    np.set_printoptions(precision=2)

    # Plot confusion matrix and save plot.
    disp = metrics.plot_confusion_matrix(clf_eval, data_test, target_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')

    # UNCOMMENT TO SET TITLE.
    disp.ax_.set_title("Normalized Confusion Matrix - " + title)
    disp.figure_.set_size_inches(9.0, 9.0, forward=True)
    plt.tight_layout()
    plt.savefig('../results/plots/cfm.png')
    plt.clf()
    plt.close()


def repl_eval(data, target, clf):
    
    # Initialize classifier.
    clf_wrapped = ClfWrap(clf)

    import pdb
    pdb.set_trace()

    # Initialize pipeline template.
    clf_pipeline = Pipeline([('scaling', StandardScaler()), ('clf', clf_wrapped)])
    clf_pipeline.fit(data, target)

    predictions = np.empty(0, dtype=int)
    while True:
        message = input()
    



if __name__ == '__main__':
    import argparse

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['rf', 'svm'], default='rf')
    parser.add_argument('--eval-method', type=str, choices=['tts', 'cv'], default='tts')
    parser.add_argument('--action', type=str, choices=['eval', 'roc', 'cm', 'repl'], default='eval')
    args = parser.parse_args()
    
    # Load data.
    data = np.load('../data/cached/data_book_relevance.npy')
    target = np.load('../data/cached/target_book_relevance.npy')

    # Perform evaluations.
    if args.method == 'rf':
        clf = RandomForestClassifier(n_estimators=100)
        clf.name = 'rf'
    elif args.method == 'svm':
        clf = SVC(gamma='auto', probability=True)
        clf.name = 'SVM'
    
    if args.action == 'eval':
        evaluate(data, target, clf, args.eval_method)
    elif args.action == 'roc':
        plot_roc(data, target, clf)
    elif args.action == 'cm':
        confusion_matrix(data, target, clf, ['No', 'Yes'], 'Random Forest', '../results/plots/cfm.png')
    elif args.action == 'repl':
        repl_eval(data, target, clf)
 

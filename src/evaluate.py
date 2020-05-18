import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import pandas as pd
from termcolor import colored

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.base import clone
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from classifiers.clf_wrap import ClfWrap
from classifiers.feat_stacking_clf import FeatureStackingClf
from classifiers.gboostclf import GradientBoostingClassifier
from feature_engineering import get_repl_processor


def save_results(category, kind, accs=None, clf_reports=None, acc_names=None, clf_reports_names=None):
    """
    Store specified results to file.

    Args:
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'.
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
        f.write('Category: {0}\n'.format(category))
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
                    f.write(clf_reports_names[idx] + ': \n')
                f.write(clf_report)
        f.write('\n')


def evaluate(data, target, category, clf, eval_method, target_names):
    """
    Evaluate model on dataset using either cross-validation or train-test split.
    Write results to file in form of classification accuracy as well as a
    classification report.

    Args:
        data (numpy.ndarray): Dataset to use (extracted features)
        target (numpy.ndarray): Labels for the samples
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'.
        clf (obj): Classification model to evaluate
        eval_method (str): Evaluation method to use. Valid values are 'tts' for train-test
        split and 'cv' for cross-validation
        target_names (list): List of labels corresponding to target values.
    """

    # Initialize pipeline template.
    clf_pipeline = Pipeline([('scaling', RobustScaler())])

    # Add classifier to pipeline.
    clf_eval = clone(clf_pipeline)
    clf_eval.steps.append(['smt', SMOTETomek()])
    clf_eval.steps.append(['clf', clf])
    
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
        
        # If predicting book relevance (binary classification problem), write fp, fn, tp and tn
        # to file.
        if category == 'book-relevance':

            # Get indices of messages representing fp, fn, tp and tn.
            idx_fail_fp = idxs_test[np.logical_and(res_pred == 1, target_test == 0)]
            idx_fail_fn = idxs_test[np.logical_and(res_pred == 0, target_test == 1)]
            idx_succ_tp = idxs_test[np.logical_and(res_pred == 0, target_test == 0)]
            idx_succ_tn = idxs_test[np.logical_and(res_pred == 1, target_test == 1)]
            
            # Save fp, fn, tp and tn messages to results folder as .xlsx files.
            sheet_raw = pd.read_excel(discussions_path)
            fp = sheet_raw.loc[idx_fail_fp, :].dropna(axis='columns').to_excel('../results/fp_' + category.replace('-', '_') + '.xlsx')
            fn = sheet_raw.loc[idx_fail_fn, :].dropna(axis='columns').to_excel('../results/fn_' + category.replace('-', '_') + '.xlsx') 
            tp = sheet_raw.loc[idx_succ_tp, :].dropna(axis='columns').to_excel('../results/tp_' + category.replace('-', '_') + '.xlsx') 
            tn = sheet_raw.loc[idx_succ_tn, :].dropna(axis='columns').to_excel('../results/tn_' + category.replace('-', '_') + '.xlsx') 

        else:
            
            # Get indices of messages representing fp, fn, tp and tn.
            idx_fail = idxs_test[res_pred != target_test]
            idx_succ = idxs_test[res_pred == target_test]
            
            # Save fp, fn, tp and tn messages to results folder as .xlsx files.
            sheet_raw = pd.read_excel(discussions_path)
            fp = sheet_raw.loc[idx_fail, :].dropna(axis='columns').to_excel('../results/fail_' + category.replace('-', '_') + '.xlsx')
            tn = sheet_raw.loc[idx_succ, :].dropna(axis='columns').to_excel('../results/success_' + category.replace('-', '_') + '.xlsx') 


        # Evaluate baseline classifiers.
        res_baseline_majority = clf_baseline_majority.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_strat = clf_baseline_strat.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_prior = clf_baseline_prior.fit(data_train, target_train).score(data_test, target_test)
        res_baseline_uniform = clf_baseline_uniform.fit(data_train, target_train).score(data_test, target_test)

        # Produce classification report.
        clf_report_eval = metrics.classification_report(target_test, res_pred, target_names=target_names)
        clf_report_baseline_majority = metrics.classification_report(target_test, clf_baseline_majority.predict(data_test), target_names=target_names)
        clf_report_baseline_uniform = metrics.classification_report(target_test, clf_baseline_uniform.predict(data_test), target_names=target_names)

        # Save results to file. 
        # Save accuracies for evaluated model, uniform baseline model and majority baseline model.
        # Save classification reports for evaluated model and uniform baseline model.
        save_results(category=category, kind='tts', accs=[res_eval, res_baseline_uniform, res_baseline_majority], 
                     clf_reports=[clf_report_eval, clf_report_baseline_uniform, clf_report_baseline_majority], 
                     acc_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'],
                     clf_reports_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'])
        

    elif eval_method == 'cv':
        # If performing cross-validation.

        # Set number of splits
        N_SPLITS = 10

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
        save_results(category=category, kind='cv', accs=[res_eval, res_baseline_uniform, res_baseline_majority], 
                     acc_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'])


def plot_roc(data, target, category, clf):
    """
    Plot ROC curve using train-test split and save results to file.
    
    Args:
        data (numpy.ndarray): Dataset to use (extracted features)
        target (numpy.ndarray): Labels for the samples
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'.
        clf (obj): Classification model to evaluate
    """
    
    # Initialize pipeline.
    clf_eval = Pipeline([('scaling', RobustScaler()), ('clf', clf)])

    # Get training and test data.
    data_train, data_test, target_train, target_test = train_test_split(data, target, shuffle=False, test_size=0.1)

    # Evaluate classifier to get probabilities.
    scores = clf_eval.fit(data_train, target_train).predict_proba(data_test)
    
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
    plt.savefig('../results/plots/roc_' + category.replace('-', '_') + '.png')
    plt.clf()
    plt.close()


def confusion_matrix(data, target, category, clf, class_names, title):
    """
    Plot and save confuction matrix for specified classifier.

    Args:
        data (numpy.ndarray): Data samples
        target (numpy.ndarray): Data labels (target variable values)
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'.
        clf (object): Classifier for which to plot the confuction matrix.
        class_names (list): List of class names
        title (str): Plot title
    """

    # Initialize random forest classifier, apply wrapper and add to pipeline.
    clf_eval = Pipeline([('scaling', RobustScaler()), ('clf', clf)])

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
    plt.savefig('../results/plots/cfm_' + category + '.png')
    plt.clf()
    plt.close()


def decompose_feature_subs_lengths(feature_subset_lenghts, lim, decomp_len):
    """
    Decompose long feature subsets into shorter partitions.

    Args:
        feature_subset_lenghts (list): List of original feature subset lengths
        lim (int): Maximum feature subset length
        decomp_len (int): Partition size

    Returns:
        (list): List of updated feature subset lengths
    """
    
    # Copy list of feature subset lengths.
    feature_subset_lenghts_copy = list(feature_subset_lenghts.copy())

    # Go over feature subset lengths.
    for idx, l in enumerate(feature_subset_lenghts_copy):

        # If next feature subset length over limit, decompose.
        if l > lim:
            num_rep = l // decomp_len
            rem = l % decomp_len
            add = num_rep * [decomp_len]
            if rem > 0:
                add += [rem]
            
            # Change previous value for decomposed value.
            feature_subset_lenghts_copy.pop(idx)
            feature_subset_lenghts_copy[idx:idx] = add

    # Return updated feature subset lengths.
    return feature_subset_lenghts_copy


def repl(clf, data_train, target_train):
    """
    Initialize and run REPL performance test.

    Args:
        clf (object): Classifier to use in REPL
        data_train (numpy.ndarray): Training data samples
        target_train (numpy.ndarray): Training data labels
    """
    
    # Initialize array for storing previous messages' features.
    hist = np.array([])
    
    # Train classifier using training data.
    clf.fit(data_train, target_train)
    
    # Get REPL processor (for converting messages to features).
    proc = get_repl_processor()
    
    # Parse initial message.
    print("### REPL evaluation ###")
    print("Type 'quit' to exit the REPL.")
    message = input('message: ')
    while message != 'quit':

        # Get features from message, add features to history.
        feat = proc(message)[np.newaxis]
        hist = np.vstack([hist, feat]) if hist.size else feat

        # Predict. The last prediction corresponds to the current message.
        res = clf.predict_proba(hist)[-1]

        # Print class probabilities.
        print(colored('P(NO)={0}'.format(res[0], end=''), 'red'))
        print(colored('P(YES)={0}'.format(res[1]), 'green'))
        message = input('message: ')


def evaluate_features(clf, f_to_name):
    """
    Evaluate features using classifiers that support feature evaluation.

    Args:
        clf (object): Trained classifier
        f_to_name (dict): Dictionary mapping feature enumerations in the
        form 'f0', 'f1', ... to their names.

    Returns:
        (dict): Dictionary mapping feature names to their estimated importances.
    """

    scores = clf.score_features(f_to_name)
    with open('../results/feature_scores.txt', 'w') as f:
        f.write('{0}:\n'.format(clf.name))
        for (name, score) in [(name, score) for name, score in \
                sorted(scores.items(), key=lambda x: x[1], reverse=True)]:
            f.write('{0}: {1:.4f}\n'.format(name, score))
        f.write('\n')


if __name__ == '__main__':
    import argparse

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['rf', 'svm', 'gboosting', 'logreg', 'stacking'], default='rf')
    parser.add_argument('--eval-method', type=str, choices=['tts', 'cv'], default='tts')
    parser.add_argument('--action', type=str, choices=['eval', 'roc', 'cm', 'repl', 'eval-features'], default='eval')
    parser.add_argument('--category', type=str, choices=['book-relevance', 'type', 'category', 'category-broad'], default='book-relevance')
    args = parser.parse_args()
    
    # Load data.
    data = np.load('../data/cached/data_' + args.category.replace('-', '_') + '.npy')
    target = np.load('../data/cached/target_' + args.category.replace('-', '_') + '.npy')

    # Load target names.
    target_names = np.load('../data/cached/target-names/target_' + args.category.replace('-', '_') + '_names.npy', allow_pickle=True)

    # Select classifier.
    if args.method == 'rf':
        clf = RandomForestClassifier(n_estimators=100)
        clf = ClfWrap(clf)
        clf.name = 'rf'
    elif args.method == 'svm':
        clf = SVC(gamma='auto', probability=True)
        clf = ClfWrap(clf)
        clf.name = 'SVM'
    elif args.method == 'logreg':
        clf = LogisticRegression(max_iter=1000)
        clf = ClfWrap(clf)
        clf.name = 'logreg'
    elif args.method == 'gboosting':
        clf = GradientBoostingClassifier()
        clf = ClfWrap(clf)
        clf.name = 'gboosting'
    elif args.method == 'stacking':
        
        # Load feature subset lengths.
        feature_subset_lengths = np.load('../data/cached/target_' + args.category.replace('-', '_') + '_feature_subset_lengths.npy')

        # Decompose long feature subsets.
        feature_subset_lengths_dec = decompose_feature_subs_lengths(feature_subset_lengths, 100, 100)
        clf = FeatureStackingClf(subset_lengths = feature_subset_lengths_dec)
        clf.name = 'stacking'
    
    # Select action.
    if args.action == 'eval':
        evaluate(data, target, args.category, clf, args.eval_method, target_names)
    elif args.action == 'roc':
        if args.category == 'book-relevance':
            plot_roc(data, target, args.category, clf)
        else:
            raise(ValueError('The ROC curve functionality can only be used for book-relevance prediction'))
    elif args.action == 'cm':
        # Set title to use.
        if args.method == 'rf':
            title = 'Random Forest'
        elif args.method == 'svm':
            title = 'Support Vector Machine'
        elif args.method == 'stacking':
            title = 'Feature Stacking'
        elif args.method == 'gboosting':
            title = 'Gradient Boosting'
        elif args.method == 'logreg':
            title = 'Logistic Regression'
        confusion_matrix(data, target, args.category, clf, target_names, title)
    elif args.action == 'repl':
        if args.category == 'book-relevance':
            repl(clf, data, target)
        else:
            raise(NotImplementedError('REPL can currently only be used for book-relevance prediction'))
    elif args.action == 'eval-features':
        clf.fit(data, target)
        with open('../data/cached/feature_names.txt', 'r') as f:
            feature_names = list(map(lambda x: x.strip(), f.readlines()))
            f_to_name = {'f' + str(idx) : feature_names[idx] for idx in range(len(feature_names))}
        evaluate_features(clf, f_to_name)
    
    sys.exit(0)
 

import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import pandas as pd
from termcolor import colored

from sklearn.model_selection import RepeatedKFold, train_test_split, GridSearchCV
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
        if accs is not None:
            for idx, acc in enumerate(accs):
                if acc_names:
                    f.write(acc_names[idx] + ': ')
                f.write(str(acc) + '\n')
        if clf_reports is not None:
            for idx, clf_report in enumerate(clf_reports):
                if clf_reports_names:
                    f.write(clf_reports_names[idx] + ': \n')
                f.write(clf_report + '\n\n')
        f.write('\n')


def sum_cr(cr1, cr2):
    """
    Sum two classification reports presented in the form of dictionaries.

    Args:
        cr1 (dict): The first classifiation report
        cr2 (dict): The second classifiation report

    Returns:
        (dict): Sum of the values in the two classification reports.
    """

    # If any classification report empty, return the other one.
    if len(cr1) == 0:
        return cr2
    if len(cr2) == 0:
        return cr1
    
    # Initialize resulting dictionary.
    res = dict()

    # Sum values.
    for key in cr1.keys():
        if key in cr2.keys():
            if key == 'accuracy':
                res[key] = cr1[key] + cr2[key]
            else:
                res[key] = {key_in : cr1[key][key_in] + cr2[key][key_in] for key_in in cr1[key].keys()}

    # Return result.
    return res


def normalize_cr(cr, val):
    """
    Normalize classification report presented in the form of a dictionary
    by dividing each value with the specified value.
    
    Args:
        cr (dict): The classification report to normalize
        val (int): The value with which to normalize.

    Returns:
        (dict): Normalize classification report in the form of a dictionary.
    """
    
    # Copy dictionary to avoid side-effects.
    res = cr.copy()

    # Go over values and normalize.
    for key in res.keys():
        if key == 'accuracy':
            res[key] /= val
        else:
            for key_in in res[key].keys():
                res[key][key_in] /= val
    
    # Return result.
    return res


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


        # Compute evaluated classifier's predictions.
        pred_eval = clf_eval.fit(data_train, target_train).predict(data_test)

        # Compute baseline classifier's predictions.
        pred_majority = clf_baseline_majority.fit(data_train, target_train).predict(data_test)
        pred_strat = clf_baseline_strat.fit(data_train, target_train).predict(data_test)
        pred_prior = clf_baseline_prior.fit(data_train, target_train).predict(data_test)
        pred_uniform = clf_baseline_uniform.fit(data_train, target_train).predict(data_test)
        
        # If predicting book relevance (binary classification problem), write fp, fn, tp and tn
        # to file.
        if category == 'book-relevance':

            # Get indices of messages representing fp, fn, tp and tn.
            idx_fail_fp = idxs_test[np.logical_and(pred_eval == 1, target_test == 0)]
            idx_fail_fn = idxs_test[np.logical_and(pred_eval == 0, target_test == 1)]
            idx_succ_tp = idxs_test[np.logical_and(pred_eval == 0, target_test == 0)]
            idx_succ_tn = idxs_test[np.logical_and(pred_eval == 1, target_test == 1)]
            
            # Save fp, fn, tp and tn messages to results folder as .xlsx files.
            sheet_raw = pd.read_excel(discussions_path)
            fp = sheet_raw.loc[idx_fail_fp, :].dropna(axis='columns').to_excel('../results/fp_' + category.replace('-', '_') + '.xlsx')
            fn = sheet_raw.loc[idx_fail_fn, :].dropna(axis='columns').to_excel('../results/fn_' + category.replace('-', '_') + '.xlsx') 
            tp = sheet_raw.loc[idx_succ_tp, :].dropna(axis='columns').to_excel('../results/tp_' + category.replace('-', '_') + '.xlsx') 
            tn = sheet_raw.loc[idx_succ_tn, :].dropna(axis='columns').to_excel('../results/tn_' + category.replace('-', '_') + '.xlsx') 

        else:
            
            # Get indices of messages representing fp, fn, tp and tn.
            idx_fail = idxs_test[pred_eval != target_test]
            idx_succ = idxs_test[pred_eval == target_test]
            
            # Save fp, fn, tp and tn messages to results folder as .xlsx files.
            sheet_raw = pd.read_excel(discussions_path)
            fp = sheet_raw.loc[idx_fail, :].dropna(axis='columns').to_excel('../results/fail_' + category.replace('-', '_') + '.xlsx')
            tn = sheet_raw.loc[idx_succ, :].dropna(axis='columns').to_excel('../results/success_' + category.replace('-', '_') + '.xlsx') 


        # Produce classification report for evaluated classifier.
        clf_report_eval = pd.DataFrame(metrics.classification_report(target_test, pred_eval, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True))
        clf_report_eval['accuracy'][1:] = ''
        cols = list(clf_report_eval.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_eval = clf_report_eval[cols]
        
        # Produce classification report for baseline majority classifier.
        clf_report_baseline_majority = pd.DataFrame(metrics.classification_report(target_test, pred_majority, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True))
        clf_report_baseline_majority['accuracy'][1:] = ''
        cols = list(clf_report_baseline_majority.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_majority = clf_report_baseline_majority[cols]
        
        # Produce classification report for baseline stratified classifier.
        clf_report_baseline_strat = pd.DataFrame(metrics.classification_report(target_test, pred_strat, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True))
        clf_report_baseline_strat['accuracy'][1:] = ''
        cols = list(clf_report_baseline_strat.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_strat = clf_report_baseline_strat[cols]
        
        # Produce classification report for baseline prior classifier.
        clf_report_baseline_prior = pd.DataFrame(metrics.classification_report(target_test, pred_prior, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True))
        clf_report_baseline_prior['accuracy'][1:] = ''
        cols = list(clf_report_baseline_prior.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_prior = clf_report_baseline_prior[cols]
        
        # Produce classification report for baseline uniform classifier.
        clf_report_baseline_uniform = pd.DataFrame(metrics.classification_report(target_test, pred_uniform, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True))
        clf_report_baseline_uniform['accuracy'][1:] = ''
        cols = list(clf_report_eval.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_uniform = clf_report_baseline_uniform[cols]

       
        # Save classification reports for evaluated model, uniform baseline model and majority baseline model.
        save_results(category=category, kind='tts', 
                     clf_reports=[clf_report_eval.to_string(), clf_report_baseline_uniform.to_string(), clf_report_baseline_majority.to_string()], 
                     clf_reports_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'])
        

    elif eval_method == 'cv':
        # If performing cross-validation.

        # Set number of splits and repeats.
        N_SPLITS = 10
        N_REPEATS = 1

        # Initialize empty classification report values accumulator.
        clf_report_eval_acc = dict()
        clf_report_baseline_majority_acc = dict()
        clf_report_baseline_strat_acc = dict()
        clf_report_baseline_prior_acc = dict()
        clf_report_baseline_uniform_acc = dict()

        # Initialize arrays for cv-fold results (for Bayesian correlated t-test).
        scores_cv_eval = np.empty(N_SPLITS*N_REPEATS, dtype=float)
        scores_cv_baseline_majority = np.empty(N_SPLITS*N_REPEATS, dtype=float)
        scores_cv_baseline_uniform = np.empty(N_SPLITS*N_REPEATS, dtype=float)

        # Initialize fold index.
        idx = 0
        for train_idx, test_idx in RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS).split(data, target):

            # Evaluate classifier.
            pred_eval = clf_eval.fit(data[train_idx, :], target[train_idx]).predict(data[test_idx, :])
            clf_report_nxt = metrics.classification_report(target[test_idx], pred_eval, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
            # scores_cv_eval[idx] = clf_report_nxt['accuracy']
            scores_cv_eval[idx] = metrics.accuracy_score(target[test_idx], pred_eval)
            clf_report_eval_acc = sum_cr(clf_report_eval_acc, clf_report_nxt)
            
            # Evaluate baseline majority classifier.
            pred_majority = clf_baseline_majority.fit(data[train_idx, :], target[train_idx]).predict(data[test_idx, :])
            clf_report_nxt = metrics.classification_report(target[test_idx], pred_majority, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
            scores_cv_baseline_majority[idx] = metrics.accuracy_score(target[test_idx], pred_majority)
            clf_report_baseline_majority_acc = sum_cr(clf_report_baseline_majority_acc, clf_report_nxt)
            
            # Evaluate baseline stratified classifier.
            pred_strat = clf_baseline_strat.fit(data[train_idx, :], target[train_idx]).predict(data[test_idx, :])
            clf_report_nxt = metrics.classification_report(target[test_idx], pred_strat, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
            clf_report_baseline_strat_acc = sum_cr(clf_report_baseline_strat_acc, clf_report_nxt)
            
            # Evaluate baseline prior classifier.
            pred_prior = clf_baseline_prior.fit(data[train_idx, :], target[train_idx]).predict(data[test_idx, :])
            clf_report_nxt = metrics.classification_report(target[test_idx], pred_prior, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
            clf_report_baseline_prior_acc = sum_cr(clf_report_baseline_prior_acc, clf_report_nxt)
            
            # Evaluate baseline uniform classifier.
            pred_uniform = clf_baseline_uniform.fit(data[train_idx, :], target[train_idx]).predict(data[test_idx, :])
            clf_report_nxt = metrics.classification_report(target[test_idx], pred_uniform, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
            scores_cv_baseline_uniform[idx] = metrics.accuracy_score(target[test_idx], pred_uniform)
            clf_report_baseline_uniform_acc = sum_cr(clf_report_baseline_uniform_acc, clf_report_nxt)
            

            # Increment fold index and print progress.
            idx += 1
            print("done {0}/{1}".format(idx, N_SPLITS*N_REPEATS))
       
        
        # Produce classification report for evaluated classifier.
        clf_report_eval = pd.DataFrame(normalize_cr(clf_report_eval_acc, N_SPLITS*N_REPEATS))
        clf_report_eval['accuracy'][1:] = ''
        cols = list(clf_report_eval.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_eval = clf_report_eval[cols]

        # Produce classification report for baseline majority classifier.
        clf_report_baseline_majority = pd.DataFrame(normalize_cr(clf_report_baseline_majority_acc, N_SPLITS*N_REPEATS))
        clf_report_baseline_majority['accuracy'][1:] = ''
        cols = list(clf_report_baseline_majority.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_majority = clf_report_baseline_majority[cols]

        # Produce classification report for baseline stratified classifier.
        clf_report_baseline_strat = pd.DataFrame(normalize_cr(clf_report_baseline_strat_acc, N_SPLITS*N_REPEATS))
        clf_report_baseline_strat['accuracy'][1:] = ''
        cols = list(clf_report_baseline_strat.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_strat = clf_report_baseline_strat[cols]

        # Produce classification report for baseline prior classifier.
        clf_report_baseline_prior = pd.DataFrame(normalize_cr(clf_report_baseline_prior_acc, N_SPLITS*N_REPEATS))
        clf_report_baseline_prior['accuracy'][1:] = ''
        cols = list(clf_report_baseline_prior.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_prior = clf_report_baseline_prior[cols]

        # Produce classification report for baseline uniform classifier.
        clf_report_baseline_uniform = pd.DataFrame(normalize_cr(clf_report_baseline_uniform_acc, N_SPLITS*N_REPEATS))
        clf_report_baseline_uniform['accuracy'][1:] = ''
        cols = list(clf_report_eval.columns)
        cols.remove('accuracy')
        cols.append('accuracy')
        clf_report_baseline_uniform = clf_report_baseline_uniform[cols]

        # Save CV results for each fold in each repetition (for Bayesian correlated t-test).
        np.save('./evaluation/data/' + category + '_' + clf_eval['clf'].name + '.npy', scores_cv_eval)
        np.save('./evaluation/data/' + category + '_' + 'majority' + '.npy', scores_cv_baseline_majority)
        np.save('./evaluation/data/' + category + '_' + 'uniform' + '.npy', scores_cv_baseline_uniform)
        
        # Save classification reports for evaluated model, uniform baseline model and majority baseline model.
        save_results(category=category, kind='cv', 
                     clf_reports=[clf_report_eval.to_string(), clf_report_baseline_uniform.to_string(), clf_report_baseline_majority.to_string()], 
                     clf_reports_names=[clf_eval['clf'].name, 'Uniform classifier', 'Majority classifier'])


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
    # plt.title('ROC curve - ')
    plt.legend(loc="lower right")
    plt.savefig('../results/plots/roc_' + category.replace('-', '_') + '_' + clf.name + '.eps')
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
    data_train, data_test, target_train, target_test = train_test_split(data, target, shuffle=False, test_size=0.1)

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
    # disp.ax_.set_title("Normalized Confusion Matrix - " + title)
    disp.figure_.set_size_inches(9.0, 9.0, forward=True)
    plt.tight_layout()
    plt.savefig('../results/plots/cfm_' + category + '_' + title.lower().replace(' ', '_') + '.eps')
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
    parser.add_argument('--method', type=str, choices=['rf', 'svm', 'gboosting', 'logreg', 'stacking'], default='rf', 
            help='classification model to use')
    parser.add_argument('--action', type=str, choices=['eval', 'roc', 'cm', 'repl', 'eval-features'], default='eval', 
            help='action to perform')
    parser.add_argument('--eval-method', type=str, choices=['tts', 'cv'], default='tts', 
            help='use train-test split or cross-validation when performing evaluation')
    parser.add_argument('--category', type=str, choices=['book-relevance', 'type', 'category', 'category-broad'], default='book-relevance', 
            help='prediction objective')
    args = parser.parse_args()
    
    # Load data.
    data = np.load('../data/cached/data_' + args.category.replace('-', '_') + '.npy')
    target = np.load('../data/cached/target_' + args.category.replace('-', '_') + '.npy')

    # Load target names.
    target_names = np.load('../data/cached/target_names/target_' + args.category.replace('-', '_') + '_names.npy', allow_pickle=True)

    
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
        clf = GradientBoostingClassifier(objective='binary:logistic') if \
                args.category == 'book-relevance' else \
                GradientBoostingClassifier(objective='multi:softmax')
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
 

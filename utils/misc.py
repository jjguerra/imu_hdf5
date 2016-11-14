import os
from output import printout
import numpy as np
from sklearn import metrics


def filename(sfile=''):

    if os.path.isdir(sfile):
        return sfile.split('/')[-1]
    elif os.path.isfile(sfile):
        return sfile
    else:
        printout(message='Wrong file name/dir passed', verbose=True, extraspaces=1)
        exit(1)


def batch(l, n):
    n = max(1, n)
    r_list = list()
    for index, i in enumerate(xrange(0, len(l), n)):
        r_list.append(l[i:i+n])

    return index, r_list


def process_confusion_matrix(confusion_matrix, row_index):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    true_pos = confusion_matrix[row_index, row_index].astype('float')  # correctly labeled as i
    false_pos = confusion_matrix[:, row_index].sum() - true_pos  # incorrectly labeled as i
    fal_neg = confusion_matrix[row_index, :].sum() - true_pos  # incorrectly labeled as non-i
    true_neg = confusion_matrix.sum().sum() - true_pos - false_pos - fal_neg
    return true_pos, false_pos, true_neg, fal_neg


def p_classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=2):
    """Build a text report showing the main classification metrics

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    digits : int
        Number of digits for formatting output floating point values

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.50      1.00      0.67         1
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.67      0.80         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.61         5
    <BLANKLINE>

    """

    labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = ['%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          labels=labels,
                                                          average=None,
                                                          sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    precission_recall = list()
    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)


    return report
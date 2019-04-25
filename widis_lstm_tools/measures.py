# -*- coding: utf-8 -*-
"""measures.py: scoring functions, performance measures, etc.


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""


def bacc(tp, tn, p, n):
    """Balanced accuracy (https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)
    
    Parameters
    ----------
    tp
        Number of true positive predictions
    tn
        Number of true negative predictions
    p
        Number of positive samples
    n
        Number of negative samples
    
    Returns
    ----------
    Balanced accuracy
    
    """
    return (tp / p + tn / n) / 2

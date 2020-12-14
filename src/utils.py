from contextlib import contextmanager
import sys
import re
import numpy as np

@contextmanager
def disable_exception_traceback():
    """
    All traceback information is suppressed and only the exception type and value are printed
    Credits: https://stackoverflow.com/questions/38598740/raising-errors-without-traceback
    """
    default_value = getattr(sys, "tracebacklimit", 1000)  # `1000` is a Python's default value
    sys.tracebacklimit = 0
    yield
    sys.tracebacklimit = default_value  # revert changes

def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    Source: NLTK
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)

def discretizeVector(x, m, M, bins):
    """
    x = vector
    m = minimum to consider for discretization
    M = maximum to consider for discretization
    bins = number of bins to discretize
    returns discretized vector (each element is in [0, bins-1])
    """
    #values < m are mapped to 0
    #values > M are mapped to bins-1
    binWidth = (M - m) / bins
    binsArr = np.arange(1, bins) * binWidth + m
    return np.digitize(x, binsArr)
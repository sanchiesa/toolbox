# tests/lreg_cv_mscore_test.py
from toolbox.lreg_cv_mscore import lreg_cv_mscore
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(1000, 5), columns=list('ABCDE'))
X = df.drop(['E'], axis=1)
y = df['E']

def test_length_of_hello_world():
    assert lreg_cv_mscore(X, y, 10) >= -1 and lreg_cv_mscore(X, y, 10) <= 1

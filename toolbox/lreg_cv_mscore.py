from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

def lreg_cv_mscore(X, y, n_cv):
    """Returns the mean score of a Linear Regression cross validation"""

    model = LinearRegression()
    cv_results = cross_validate(model, X, y, cv=n_cv)

    return cv_results['test_score'].mean()


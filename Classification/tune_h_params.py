from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

KF = KFold(5)
SCORING = {
           # 'accuracy': make_scorer(accuracy_score),
           # 'precision': make_scorer(precision_score),
           # 'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}


# LogisticRegression
# -------------------------------------------------------------------------------------------
def optimize_log_reg(trial, x, y):
    C = trial.suggest_float('C', 1.0, 30.0, step=0.1)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    max_iter = trial.suggest_int('max_iter', 50, 300)

    model = make_pipeline(RobustScaler(),
                          LogisticRegression(C=C,
                                             solver=solver,
                                             max_iter=max_iter))
    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# SVC
# -------------------------------------------------------------------------------------------
def optimize_svc(trial, x, y):
    C = trial.suggest_float('C', 1.0, 30.0, step=0.1)
    gamma = trial.suggest_float('gamma', 0.000001, 0.0001)

    model = make_pipeline(RobustScaler(),
                          SVC(C=C,
                              gamma=gamma,
                              random_state=1))
    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# KNeighborsClassifier
# -------------------------------------------------------------------------------------------
def optimize_knn(trial, x, y):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 10)
    leaf_size = trial.suggest_int('leaf_size', 10, 60)

    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                 leaf_size=leaf_size)
    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# DecisionTreeClassifier
# -------------------------------------------------------------------------------------------
def optimize_d_tree(trial, x, y):
    max_depth = trial.suggest_int('max_depth', 4, 15)

    model = DecisionTreeClassifier(max_depth=max_depth,
                                   random_state=1)
    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# RandomForestClassifier
# -------------------------------------------------------------------------------------------
def optimize_r_forest(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 200, 3500, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   random_state=1)

    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# GradientBoostingClassifier
# -------------------------------------------------------------------------------------------
def optimize_g_boost(trial, x, y):
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 200, 3500, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=1)

    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# XGBClassifier
# -------------------------------------------------------------------------------------------
def optimize_xg_boost(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 200, 3500, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)

    model = XGBClassifier(n_estimators=n_estimators,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          random_state=1)

    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# LGBMClassifier
# -------------------------------------------------------------------------------------------
def optimize_lg_boost(trial, x, y):
    num_leaves = trial.suggest_int('num_leaves', 5, 40)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 200, 3500, step=100)
    max_bin = trial.suggest_int('max_bin', 30, 250)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)

    model = LGBMClassifier(num_leaves=num_leaves,
                           max_depth=max_depth,
                           learning_rate=learning_rate,
                           n_estimators=n_estimators,
                           max_bin=max_bin,
                           feature_fraction=feature_fraction,
                           bagging_fraction=bagging_fraction,
                           feature_fraction_seed=1,
                           bagging_seed=1)

    cv_results = cross_validate(estimator=model, X=x, y=y, cv=KF, scoring=SCORING, n_jobs=-1)
    return cv_results['test_f1_score'].mean()


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass

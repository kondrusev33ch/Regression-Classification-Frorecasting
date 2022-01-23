from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, TweedieRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

KF = KFold(10)


# Ridge
# -------------------------------------------------------------------------------------------
def optimize_ridge(trial, x, y):
    alpha = trial.suggest_float('alpha', 1.0, 30.0)

    model = make_pipeline(RobustScaler(), Ridge(alpha=alpha))
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)
    return -cv_results['test_score'].mean()


# Lasso
# -------------------------------------------------------------------------------------------
def optimize_lasso(trial, x, y):
    alpha = trial.suggest_float('alpha', 0.0001, 0.0007)

    model = make_pipeline(RobustScaler(), Lasso(alpha=alpha, random_state=13))
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)
    return -cv_results['test_score'].mean()


# Elastic-Net
# -------------------------------------------------------------------------------------------
def optimize_elastic_net(trial, x, y):
    alpha = trial.suggest_float('alpha', 0.0001, 0.0007)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    model = make_pipeline(RobustScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=13))
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)
    return -cv_results['test_score'].mean()


# SVR
# -------------------------------------------------------------------------------------------
def optimize_svr(trial, x, y):
    C = trial.suggest_float('C', 1.0, 30.0)
    epsilon = trial.suggest_float('epsilon', 0.0001, 0.01)
    tol = trial.suggest_float('tol', 0.000001, 0.001)
    gamma = trial.suggest_float('gamma', 0.000001, 0.0001)

    model = make_pipeline(RobustScaler(), SVR(C=C, epsilon=epsilon, tol=tol, gamma=gamma))
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)
    return -cv_results['test_score'].mean()


# GBR
# -------------------------------------------------------------------------------------------
def optimize_gbr(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 300, 3500, step=100)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 30)

    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      max_features='sqrt',
                                      min_samples_leaf=min_samples_leaf,
                                      loss='huber',
                                      random_state=13)

    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)

    return -cv_results['test_score'].mean()


# LGBMR
# -------------------------------------------------------------------------------------------
def optimize_lgbmr(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 300, 3500, step=100)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    num_leaves = trial.suggest_int('num_leaves', 5, 30)
    max_bin = trial.suggest_int('max_bin', 50, 250)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 0.99)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 0.99)

    model = LGBMRegressor(objective='regression',
                          n_estimators=n_estimators,
                          num_leaves=num_leaves,
                          learning_rate=learning_rate,
                          max_depth=max_depth,
                          max_bin=max_bin,
                          feature_fraction_seed=13,
                          feature_fraction=feature_fraction,
                          bagging_seed=13,
                          bagging_fraction=bagging_fraction)
    
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)

    return -cv_results['test_score'].mean()


# XGBR
# -------------------------------------------------------------------------------------------
def optimize_xgbr(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 300, 3500, step=100)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 0.99)
    subsample = trial.suggest_float('subsample', 0.1, 0.99)

    model = XGBRegressor(learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_child_weight=0,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         nthread=-1,
                         random_state=13)
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)

    return -cv_results['test_score'].mean()


# Tweedie
# -------------------------------------------------------------------------------------------
def optimize_tweedie(trial, x, y):
    alpha = trial.suggest_float('alpha', 0.0001, 1.0)

    model = make_pipeline(RobustScaler(), TweedieRegressor(alpha=alpha))
    cv_results = cross_validate(model, x, y, cv=KF, scoring='neg_root_mean_squared_error',
                                return_train_score=True, n_jobs=-1)
    return -cv_results['test_score'].mean()


# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass

"""
Competition:
    https://www.kaggle.com/c/home-data-for-ml-course

References:
    https://www.kaggle.com/cheesu/house-prices-1st-approach-to-data-science-process/notebook
    https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking/notebook
"""

import numpy as np
import pandas as pd
from time import time
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, TweedieRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

import helpers as h


# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_full = pd.read_csv('data/train.csv', index_col='Id')
    test_full = pd.read_csv('data/test.csv', index_col='Id')

    print('Train set shape:', train_full.shape)  # Train set shape: (1460, 80)
    print('Test set shape:', test_full.shape)  # Test set shape: (1459, 79)

    # Create variables we will working with
    X_train = train_full.copy()
    X_test = test_full.copy()
    y = X_train.pop('SalePrice')

    # Dealing with missing data
    # =========================================================================================
    # First of all lets look at missing values in our target column
    print(sum(y.isna()))  # 0
    # Luckily we do not have missing values in target column

    # Now lets see what is going on in X_train and X_test
    train_missing = X_train.isnull().sum().sort_values(ascending=False)
    train_missing = train_missing[train_missing != 0]  # throw away rows with zeros
    test_missing = X_test.isnull().sum().sort_values(ascending=False)
    test_missing = test_missing[test_missing != 0]

    missing = pd.concat([train_missing, test_missing], axis=1, keys=['Train', 'Test'])
    print(missing)
    #                Train    Test
    # PoolQC        1453.0  1456.0 -> Na
    # MiscFeature   1406.0  1408.0 -> Na
    # Alley         1369.0  1352.0 -> Na
    # Fence         1179.0  1169.0 -> Na
    # FireplaceQu    690.0   730.0 -> Na
    # LotFrontage    259.0   227.0 -> ?
    # GarageType      81.0    76.0 -> Na
    # GarageYrBlt     81.0    78.0 -> 0
    # GarageQual      81.0    78.0 -> Na
    # GarageCond      81.0    78.0 -> Na
    # GarageFinish    81.0    78.0 -> Na
    # BsmtFinType2    38.0    42.0 -> Na
    # BsmtExposure    38.0    44.0 -> Na
    # BsmtCond        37.0    45.0 -> Na
    # BsmtFinType1    37.0    42.0 -> Na
    # BsmtQual        37.0    44.0 -> Na
    # MasVnrArea       8.0    15.0 -> 0
    # MasVnrType       8.0    16.0 -> Na
    # Electrical       1.0     NaN -> mode
    # MSZoning         NaN     4.0 -> ?
    # Functional       NaN     2.0 -> mode
    # BsmtHalfBath     NaN     2.0 -> 0
    # BsmtFullBath     NaN     2.0 -> 0
    # Utilities        NaN     2.0 -> mode
    # KitchenQual      NaN     1.0 -> mode
    # SaleType         NaN     1.0 -> mode
    # BsmtFinSF1       NaN     1.0 -> 0
    # GarageCars       NaN     1.0 -> 0
    # BsmtUnfSF        NaN     1.0 -> 0
    # TotalBsmtSF      NaN     1.0 -> 0
    # Exterior2nd      NaN     1.0 -> mode
    # Exterior1st      NaN     1.0 -> mode
    # GarageArea       NaN     1.0 -> 0
    # BsmtFinSF2       NaN     1.0 -> 0

    # Since we read data descriptions, it is pretty easy to deal with these missing data
    # But there are some problems with LotFrontage and MSZoning, it is not clear what we
    # should fill them with.

    fill_with_Na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
                    'GarageQual', 'GarageCond', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure',
                    'BsmtCond', 'BsmtFinType1', 'BsmtQual', 'MasVnrType']
    fill_with_0 = ['GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',
                   'GarageCars', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF2']
    fill_with_mode = ['Electrical', 'Functional', 'Utilities', 'KitchenQual',
                      'SaleType', 'Exterior2nd', 'Exterior1st']
    unknown = ['LotFrontage', 'MSZoning']

    # Check we did not forget something
    assert missing.shape[0] == len(fill_with_Na) + len(fill_with_0) + len(fill_with_mode) + len(unknown)

    # Replace missing values inplace for X_train and X_test sets
    for col in fill_with_Na:
        X_train[col].replace(np.nan, 'Na', inplace=True)
        X_test[col].replace(np.nan, 'Na', inplace=True)

    for col in fill_with_0:
        X_train[col].replace(np.nan, 0, inplace=True)
        X_test[col].replace(np.nan, 0, inplace=True)

    for col in fill_with_mode:
        X_train[col].replace(np.nan, X_train[col].mode()[0], inplace=True)
        X_test[col].replace(np.nan, X_test[col].mode()[0], inplace=True)

    # I will use idea from ERTUĞRUL DEMIR, reference on top:
    # Logic and bright idea to .groupby() with correlated column and fill them with most common
    # type in this group or median. Not just drop these columns.
    X_train['LotFrontage'] = X_train.groupby(
        ['Neighborhood'])['LotFrontage'].apply(lambda a: a.fillna(a.median()))
    X_test['LotFrontage'] = X_test.groupby(
        ['Neighborhood'])['LotFrontage'].apply(lambda a: a.fillna(a.median()))

    # No need to do it for X_train, because missing values are only in X_test
    X_test['MSZoning'] = X_test.groupby(
        ['MSSubClass'])['MSZoning'].apply(lambda a: a.fillna(a.mode()[0]))

    # Now we shouldn't have any missing values
    assert X_train.isna().sum().sum() + X_test.isna().sum().sum() == 0, '[!] Missing values detected v1'

    # There are columns with no sense being numerical, so lets convert them into str columns
    for col in ['MSSubClass', 'YrSold', 'MoSold']:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Analyse numeric data
    # =========================================================================================
    train_num_attributes = train_full.select_dtypes(exclude=['object']).drop('SalePrice',
                                                                             axis=1).columns
    h.plot_numerical_attributes(train_full, 'SalePrice', train_num_attributes)
    # By analysing these plots, we can easily find outliers and follow the trend of SalePrice
    # correlation

    # By watching on correlation curves we can spot some useless features
    to_drop_later = ['3SsnPorch', 'LowQualFinSF', 'PoolArea']
    to_drop = ['MiscVal', 'MoSold', 'YrSold', 'BsmtFullBath', 'BsmtHalfBath',
               'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'MSSubClass']

    # Drop useless features
    X_train.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)

    # Since we dropped some useless columns, we need to update the list of numerical columns
    actual_train_num_attributes = [a for a in train_num_attributes if a not in to_drop + to_drop_later]

    # Finding outliers with IsolationForest
    outliers_indices = set()
    for col in actual_train_num_attributes:
        temp = X_train[col].to_numpy().reshape(len(X_train[col]), 1)  # prepare column data
        isolation = IsolationForest(random_state=1, contamination=0.002).fit(temp)  # 0.002 fits best
        predictions = isolation.predict(temp)  # finds predictions

        # Get outliers indices                                        pay attention to start=1!
        outliers_indices.update([i for i, x in enumerate(predictions, start=1) if x == -1])

    # Drop rows with outliers
    if outliers_indices:
        X_train = X_train.drop(outliers_indices)
        y = y.drop(outliers_indices)

    # Find skewed data
    h.histplot_numerical_attributes(train_full, actual_train_num_attributes + ['SalePrice'])
    skewed = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
              'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
              'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
    # We will deal with skewed data later

    # Analyse categorical data
    # =========================================================================================
    train_cat_attributes = train_full.select_dtypes(include=['object']).columns

    print('Pay attention to this columns:')
    for col in train_cat_attributes:
        maximum = X_train[col].value_counts().max()
        if maximum > len(X_train) * 0.95:  # one label met more than 95%
            print(col, maximum)
            # Street 1431 -> [+] we can see logic correlation
            # Utilities 1434 -> [-] almost everything has one label
            # LandSlope 1368 -> [-] week correlation, and huge number in one label
            # Condition2 1422 -> [+] good correlation, even that we have huge number, this is a good feature
            # RoofMatl 1416 -> [+] good correlation, as in previous case
            # Heating 1404 -> [+] meh, ok, let it be
            # PoolQC 1430 -> [-] almost all pools in excellent condition, this will not help us much
            # MiscFeature 1384 -> [+] useful, good correlation

    h.plot_categorical_attributes(train_full, 'SalePrice', train_cat_attributes)

    # One more suspicious feature:
    # GarageQual -> [-] there is no good price dependency on garage quality

    # Drop few more useless columns
    to_drop = ['Utilities', 'LandSlope', 'PoolQC', 'GarageQual']
    X_train.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)

    # After visual analyse of our categorical data we should translate it to numerical data
    # One-hot-encoder is good for unordered features and ordinal encoder is good for ordered
    # features
    # Lets start with ordered: Quality, Condition, ...
    train_cat_attributes = X_train.select_dtypes(include=['object']).columns

    # ord_columns = ['HouseStyle', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    #                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
    #                'KitchenQual', 'Functional', 'GarageFinish', 'GarageCond']

    map_HouseStyle = {'1Story': 1, '1.5Unf': 2, '1.5Fin': 3, '2Story': 4,
                      '2.5Unf': 5, '2.5Fin': 6, 'SFoyer': 7, 'SLvl': 8}
    X_train['HouseStyle'] = X_train['HouseStyle'].map(map_HouseStyle).astype(int)
    X_test['HouseStyle'] = X_test['HouseStyle'].map(map_HouseStyle).astype(int)

    map_Qual_and_Cond = {'Na': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageCond']:
        X_train[col] = X_train[col].map(map_Qual_and_Cond).astype(int)
        X_test[col] = X_test[col].map(map_Qual_and_Cond).astype(int)

    map_BsmtExposure = {'Na': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    X_train['BsmtExposure'] = X_train['BsmtExposure'].map(map_BsmtExposure).astype(int)
    X_test['BsmtExposure'] = X_test['BsmtExposure'].map(map_BsmtExposure).astype(int)

    map_BsmtFinTipes = {'Na': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        X_train[col] = X_train[col].map(map_BsmtFinTipes).astype(int)
        X_test[col] = X_test[col].map(map_BsmtFinTipes).astype(int)

    map_Functional = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
    X_train['Functional'] = X_train['Functional'].map(map_Functional).astype(int)
    X_test['Functional'] = X_test['Functional'].map(map_Functional).astype(int)

    map_GarageFinish = {'Na': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    X_train['GarageFinish'] = X_train['GarageFinish'].map(map_GarageFinish).astype(int)
    X_test['GarageFinish'] = X_test['GarageFinish'].map(map_GarageFinish).astype(int)

    # Since neighborhood has good correlation with target, it is better to set them manually
    map_Neighborhood = {'MeadowV': 1, 'IDOTRR': 1, 'BrDale': 1, 'BrkSide': 2, 'OldTown': 2,
                        'Edwards': 2, 'Sawyer': 3, 'Blueste': 3, 'SWISU': 3, 'NPkVill': 3,
                        'NAmes': 3, 'Mitchel': 4, 'SawyerW': 5, 'NWAmes': 5, 'Gilbert': 5,
                        'Blmngtn': 5, 'CollgCr': 5, 'ClearCr': 6, 'Crawfor': 6, 'Veenker': 7,
                        'Somerst': 7, 'Timber': 8, 'StoneBr': 9, 'NridgHt': 10, 'NoRidge': 10}
    X_train['Neighborhood'] = X_train['Neighborhood'].map(map_Neighborhood).astype(int)
    X_test['Neighborhood'] = X_test['Neighborhood'].map(map_Neighborhood).astype(int)

    # One-hot-encoding with .get_dummies
    # Since I want to create new features, and do it with one data set is much easier, I need
    # to concat data sets. Must concat datasets before get_dummies, because training and testing
    # sets might have different labels in object type columns
    X = pd.concat([X_train, X_test])
    X = pd.get_dummies(data=X)

    # Check
    assert X.select_dtypes(include=['object']).empty, '[!] Object type columns still exists'
    assert X.isna().sum().sum() == 0, '[!] Missing values detected after One-hot'

    # Create New Features
    # =========================================================================================
    X['TotalSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF']
    X['TotalPorchSF'] = (X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch']
                         + X['ScreenPorch'] + X['WoodDeckSF'])
    X['YearBlRm'] = X['YearBuilt'] + X['YearRemodAdd']

    # Merging quality and conditions.
    X['TotalExtQual'] = X['ExterQual'] + X['ExterCond']
    X['TotalBsmtQual'] = X['BsmtQual'] + X['BsmtCond'] + X['BsmtFinType1'] + X['BsmtFinType2']
    X['TotalQual'] = (X['OverallQual'] + X['TotalExtQual'] + X['GarageCond']
                      + X['TotalBsmtQual'] + X['KitchenQual'] + X['HeatingQC'])

    # Creating new features by using new quality indicators.
    X['QualGr'] = X['TotalQual'] * X['GrLivArea']
    X['QualBsm'] = X['TotalBsmtQual'] * (X['BsmtFinSF1'] + X['BsmtFinSF2'])
    X['QualPorch'] = X['TotalExtQual'] * X['TotalPorchSF']
    X['QualExt'] = X['TotalExtQual'] * X['MasVnrArea']
    X['QualGrg'] = X['BsmtCond'] * X['GarageArea']
    X['QlLivArea'] = (X['GrLivArea'] - X['LowQualFinSF']) * X['TotalQual']
    X['QualSFNg'] = X['QualGr'] * X['Neighborhood']

    # Creating some simple features.
    X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    X['Has2ndFloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasGarage'] = X['BsmtCond'].apply(lambda x: 1 if x > 0 else 0)
    X['HasBsmt'] = X['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
    X['HasFireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    X['HasPorch'] = X['QualPorch'].apply(lambda x: 1 if x > 0 else 0)

    new_features = ['TotalSF', 'TotalPorchSF', 'YearBlRm', 'TotalExtQual',
                    'TotalBsmtQual', 'TotalQual', 'QualGr', 'QualBsm',
                    'QualPorch', 'QualExt', 'QualGrg', 'QlLivArea', 'QualSFNg']

    # Drop columns saved for creating new features
    X.drop(columns=to_drop_later, inplace=True)

    h.plot_numerical_attributes(X.join(y), 'SalePrice', new_features)

    assert X.isna().sum().sum() == 0, '[!] Missing values detected after new features creation'

    # Transforming the Data
    # =========================================================================================
    # First we should transform skewed data with boxcox
    skew_features = np.abs(X[skewed].apply(lambda a: skew(a)).sort_values(ascending=False))

    high_skew = skew_features[skew_features > 0.3]
    skew_index = high_skew.index

    for i in skew_index:
        X[i] = boxcox1p(X[i], boxcox_normmax(X[i] + 1))

    # Separate training set and testing set
    X_train = X.iloc[:len(X_train), :]
    X_test = X.iloc[len(X_train):, :]

    # Plot final correlations
    h.plot_correlations(X_train.join(y), 'SalePrice')

    # Our target is skewed, so we should apply log on it
    h.plot_dist3(X_train.join(y), 'SalePrice')
    y = np.log1p(y)
    h.plot_dist3(X_train.join(y), 'SalePrice')

    # Modeling
    # =========================================================================================

    # Ridge Regression
    # ----------------
    ridge = make_pipeline(RobustScaler(),
                          Ridge(alpha=25.55))

    # Lasso Regression
    # ----------------
    lasso = make_pipeline(RobustScaler(),
                          Lasso(alpha=0.0003943, random_state=13))

    # Elastic-Net Regression
    # ----------------------
    elastic_net = make_pipeline(RobustScaler(),
                                ElasticNet(random_state=13, alpha=0.0003939, l1_ratio=0.99881))

    # SVR
    # ---
    svr = make_pipeline(RobustScaler(),
                        SVR(C=27.26349, epsilon=0.006555, gamma=9.995488276825254e-05, tol=0.0009))

    # Gradient Boosting
    # -----------------
    gbr = GradientBoostingRegressor(n_estimators=3400, learning_rate=0.00676, max_depth=3,
                                    max_features='sqrt', min_samples_leaf=16, loss='huber',
                                    random_state=13)

    # LightGBM
    # --------
    lgbmr = LGBMRegressor(objective='regression', n_estimators=3200, num_leaves=18, max_depth=3,
                          learning_rate=0.006797, max_bin=108, bagging_fraction=0.71368,
                          n_jobs=-1, bagging_seed=13, feature_fraction_seed=13, feature_fraction=0.11954)

    # Extreme Gradient Boosting
    # -------------------------
    xgbr = XGBRegressor(learning_rate=0.01206, n_estimators=2300, max_depth=4, min_child_weight=0,
                        subsample=0.28091, colsample_bytree=0.13047, nthread=-1, random_state=13)

    # Tweedie Regression
    # -----------------
    tweedie = make_pipeline(RobustScaler(),
                            TweedieRegressor(alpha=0.0222))

    # Stacking Regression
    # ------------------
    stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elastic_net, svr,
                                                gbr, lgbmr, xgbr, tweedie),
                                    meta_regressor=xgbr,
                                    use_features_in_secondary=True)

    # Parameters tuning example
    # =========================================================================================
    # import optuna
    # import tune_hyperparameters
    # from functools import partial
    #
    # study = optuna.create_study(direction='minimize')
    # ridge_optimization_fu = partial(tune_hyperparameters.optimize_ridge, x=X_train, y=y)
    # study.optimize(ridge_optimization_fu, n_trials=200)
    # print(study.best_params)
    # print(study.best_value)
    # {'alpha': 25.55002110067531}
    # 0.10926311682714433

    # Cross Validation
    # =========================================================================================
    estimators = [ridge, lasso, elastic_net, svr, gbr, lgbmr, xgbr, tweedie]
    labels = ['Ridge', 'Lasso', 'Elastic-Net', 'SVR', 'GBR', 'LGBMR', 'XGBR', 'Tweedie']

    kf = KFold(10)
    print(h.model_check(X_train, y, estimators, labels, kf))
    #           Name  Train RMSE  Test RMSE  Test Std
    # 4          GBR    0.080829   0.106824  0.012348
    # 1        Lasso    0.099008   0.107810  0.010995
    # 2  Elastic-Net    0.099001   0.107810  0.010996
    # 6         XGBR    0.049282   0.108163  0.013845
    # 5        LGBMR    0.076061   0.108475  0.013179
    # 7      Tweedie    0.100378   0.109262  0.010260
    # 0        Ridge    0.100077   0.109263  0.010316
    # 3          SVR    0.100494   0.109791  0.010561

    # Stacking and Blending
    # =========================================================================================
    # Fit the models
    stack_gen = stack_gen.fit(X_train.values, y.values)
    for estimator in estimators:
        estimator.fit(X_train, y)

    # Blend the models
    def blend_models_predict(x: pd.DataFrame) -> pd.Series:
        return (0.1 * ridge.predict(x) +
                0.1 * lasso.predict(x) +
                0.1 * elastic_net.predict(x) +
                0.1 * svr.predict(x) +
                0.2 * gbr.predict(x) +
                0.1 * lgbmr.predict(x) +
                0.05 * xgbr.predict(x) +
                0.05 * tweedie.predict(x) +
                0.2 * stack_gen.predict(x.values))


    # Submission
    # =========================================================================================
    submission = pd.read_csv('data/test.csv')

    # Inversing and flooring log scaled sale price predictions
    submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(X_test)))

    # Defining outlier quartile ranges
    q1 = submission['SalePrice'].quantile(0.0050)
    q2 = submission['SalePrice'].quantile(0.99)

    # Applying weights to outlier ranges to smooth them
    submission['SalePrice'] = submission['SalePrice'].apply(lambda a: a if a > q1 else a * 0.77)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda a: a if a < q2 else a * 1.1)

    submission = submission[['Id', 'SalePrice']]
    submission.to_csv(f'submissions/submission_{time()}.csv', index=False)

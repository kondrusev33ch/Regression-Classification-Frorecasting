from helpers import *
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate


# ---------------------------------------------------------------------------------------------
def model_check(x: pd.DataFrame, y_: pd.DataFrame, models: list[tuple]):
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    k_fold = KFold(5)
    model_table = pd.DataFrame()
    index = 0

    for name, model in models:
        model_table.loc[index, 'Name'] = name
        cv_results = cross_validate(estimator=model, X=x, y=y_, cv=k_fold, scoring=scoring, n_jobs=-1)
        model_table.loc[index, 'Accuracy'] = cv_results['test_accuracy'].mean()
        model_table.loc[index, 'Precision'] = cv_results['test_precision'].mean()
        model_table.loc[index, 'Recall'] = cv_results['test_recall'].mean()
        model_table.loc[index, 'F1_score'] = cv_results['test_f1_score'].mean()

        index += 1

    return model_table.sort_values(by=['F1_score'])


# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
    dataset_full = pd.read_csv('data/heart.csv')

    # Analyse data
    # =========================================================================================
    print(dataset_full.info())

    # RangeIndex: 918 entries, 0 to 917
    # Data columns (total 12 columns):
    #  #   Column          Non-Null Count  Dtype
    # ---  ------          --------------  -----
    #  0   Age             918 non-null    int64
    #  1   Sex             918 non-null    object
    #  2   ChestPainType   918 non-null    object
    #  3   RestingBP       918 non-null    int64
    #  4   Cholesterol     918 non-null    int64
    #  5   FastingBS       918 non-null    int64
    #  6   RestingECG      918 non-null    object
    #  7   MaxHR           918 non-null    int64
    #  8   ExerciseAngina  918 non-null    object
    #  9   Oldpeak         918 non-null    float64
    #  10  ST_Slope        918 non-null    object
    #  11  HeartDisease    918 non-null    int64
    # dtypes: float64(1), int64(6), object(5)

    # No missing values, nice
    # Our target is HeartDisease

    # Now lets check our data on misspells
    # ------------------------------------
    categorical_features = dataset_full.select_dtypes(include=['object']).columns.tolist()
    for cat_col in categorical_features:
        print(cat_col, f'{dataset_full[cat_col].unique()}')
        # Sex ['M' 'F']
        # ChestPainType ['ATA' 'NAP' 'ASY' 'TA']
        # RestingECG ['Normal' 'ST' 'LVH']
        # ExerciseAngina ['N' 'Y']
        # ST_Slope ['Up' 'Flat' 'Down']

        # After compare with documentations, everything seems correct

    # Analyse numeric features
    # =========================================================================================
    # Correlation to target
    # ---------------------
    # Visual analysis
    numerical_correlations(dataset_full)

    numeric_features = dataset_full.select_dtypes(exclude=['object']).drop(['HeartDisease'],
                                                                           axis=1).columns
    plot_numerical_features(dataset_full, 'HeartDisease', numeric_features)

    # FastingBS should be treated as categorical
    categorical_features.append('FastingBS')

    # Week correlation, RestingBP almost do not have correlation to the target,
    # so lets drop it
    to_drop = ['RestingBP']

    # Outliers
    # --------
    # No outliers

    # Skewed data
    # -----------
    # Update numeric_features
    numeric_features = dataset_full.select_dtypes(exclude=['object']).drop(['HeartDisease',
                                                                            'FastingBS',
                                                                            'RestingBP'],
                                                                           axis=1).columns
    # Visual analysis
    histplot_numerical_features(dataset_full, numeric_features)

    # Oldpeak is skewed to much, others are pretty good
    skewed = ['Oldpeak']

    # Analyse categorical features
    # =========================================================================================
    # First lets check categorical features for highly dominant labels
    for col in categorical_features:
        maximum = dataset_full[col].value_counts().max()
        if maximum > len(dataset_full) * 0.95:  # one label met more than 95%
            print(col, maximum)
            # Nothing found

    # Now we can plot them
    plot_categorical_features(dataset_full, 'HeartDisease', categorical_features)
    # What I can say after analyze graphs:
    #   Sex             it is more likely for males to have a heart disease
    #
    #   ChestPainType   with ASY chances to have heart disease are more than 75%,
    #                   for TA pain is about fifty fifty and for other types less than 40%.
    #                   In most cases heart disease are asymptomatic
    #                   To get more precise percents:
    #                       dataset_full.groupby('ChestPainType')['HeartDisease'].mean()
    #
    #   RestingECG      about fifty fifty with normal resting electrocardiogram results, 70%
    #                   for ST and 60% for LVH.
    #                   There is no strong dependency, slightly more chance to have heart disease
    #                   is person has ST or LVH
    #
    #   ExerciseAngina  it is more likely to have heart disease for person with exercise angina,
    #                   about 90%
    #
    #   St_Slope        low chances for Up, 80% for Flat and about 80% for Down, which means
    #                   if person has Flat or Down slope of the peak exercise ST segment, he is
    #                   most likely in trouble
    #
    #   FastingBS       about 80% to have heart disease if fasting blood sugar higher than 120,
    #                   fifty fifty otherwise

    # Conclusion: All columns will be useful for us, there are no columns without target
    # dependency, no columns with highly dominant labels. No columns to drop.

    # Preprocessing
    # =========================================================================================
    X = dataset_full.copy()
    y = X.pop('HeartDisease')

    # Drop useless columns
    X.drop(to_drop, axis=1, inplace=True)  # dropping this column increased the final F1 score

    # Dealing with skewed data
    # Since our skewed columns contains negative values, PowerTransformer with default
    # params will be good choice
    pt = PowerTransformer()  # default=’yeo-johnson’
    for col in skewed:
        np_X = X[col].to_numpy()
        X[col] = pt.fit_transform(np_X.reshape((len(np_X), 1)))
        # Unfortunately it didn't help much with Oldpeak column

    # Convert categorical columns to numerical
    # We can simply convert Yes No and Male Female to 1 0 format
    X.Sex = (X.Sex == 'M').astype(int)
    X.ExerciseAngina = (X.ExerciseAngina == 'Y').astype(int)

    # For the rest columns we apply get_dummies
    X = pd.get_dummies(data=X)

    # New features?
    # No, I do not know what new features to add

    # Last check before splitting dataset into training and testing
    assert X.isna().sum().sum() == 0, '[!] Missing values detected'
    assert X.select_dtypes(include=['object']).empty, '[!] Object type columns still exists'
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Modelling
    # =========================================================================================
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    # 10 models to test, cool
    log_reg = make_pipeline(RobustScaler(),
                            LogisticRegression(C=1.2,
                                               solver='lbfgs',
                                               max_iter=273))
    # {'C': 1.2, 'solver': 'lbfgs', 'max_iter': 273}
    # F1: 0.86150981004413

    gaus = make_pipeline(RobustScaler(),
                         GaussianNB())

    svc = make_pipeline(RobustScaler(),
                        SVC(C=19.1,
                            gamma=0.00008996,
                            random_state=1))
    # {'C': 19.1, 'gamma': 8.996062616284533e-05}
    # F1: 0.8578767752432569

    knn = KNeighborsClassifier(n_neighbors=7,
                               leaf_size=14)
    # {'n_neighbors': 7, 'leaf_size': 14}
    # F1: 0.717497570588803

    d_tree = DecisionTreeClassifier(max_depth=5,
                                    random_state=1)
    # {'max_depth': 5}
    # F1: 0.8451172574292874

    r_forest = RandomForestClassifier(n_estimators=2300,
                                      max_depth=4,
                                      random_state=1)
    # {'n_estimators': 2300, 'max_depth': 4}
    # F1: 0.8766468220413209

    g_boost = GradientBoostingClassifier(learning_rate=0.00418,
                                         n_estimators=1500,
                                         max_depth=3,
                                         random_state=1)
    # {'learning_rate': 0.004180227337998437, 'n_estimators': 1500, 'max_depth': 3}
    # F1: 0.8725469893916882

    xg_boost = XGBClassifier(n_estimators=1300,
                             max_depth=3,
                             learning_rate=0.007135,
                             random_state=1)
    # {'n_estimators': 1300, 'max_depth': 3, 'learning_rate': 0.007135229781758866}
    # F1: 0.8766011404953653

    lg_boost = LGBMClassifier(num_leaves=30,
                              max_depth=3,
                              learning_rate=0.0393,
                              n_estimators=300,
                              max_bin=231,
                              feature_fraction=0.2629,
                              bagging_fraction=0.9417,
                              feature_fraction_seed=1,
                              bagging_seed=1)
    # {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.03929960234775651,
    # 'n_estimators': 300, 'max_bin': 231, 'feature_fraction': 0.2629002599655336,
    # 'bagging_fraction': 0.94174328133247}
    # 0.8922727272727272

    estimators = [('log_reg', log_reg),
                  ('gaus', gaus),
                  ('svc', svc),
                  ('knn', knn),
                  ('d_tree', d_tree),
                  ('r_forest', r_forest),
                  ('g_boost', g_boost),
                  ('xg_boost', xg_boost),
                  ('lg_boost', lg_boost)]
    s_cls = StackingClassifier(estimators=estimators, final_estimator=g_boost)

    # Example how I tuned models hyper parameters
    # -------------------------------------------
    # import optuna
    # import tune_h_params
    # from functools import partial
    #
    # study = optuna.create_study(direction='maximize')
    # g_boost_optimization_fu = partial(tune_h_params.optimize_g_boost, x=X_train, y=y_train)
    # study.optimize(g_boost_optimization_fu, n_trials=100)
    # print(study.best_params)  # => {'learning_rate': 0.00418, 'n_estimators': 1500, 'max_depth': 3}
    # print(study.best_value)  # => 0.8725469893916882

    # Cross Validation
    # =========================================================================================
    print(model_check(X_train, y_train, estimators))
    #        Name  Accuracy  Precision    Recall  F1_score
    # 3       knn  0.687988   0.706659  0.734941  0.717498
    # 1      gaus  0.824248   0.836688  0.841245  0.837633
    # 4    d_tree  0.829652   0.831190  0.860847  0.845117
    # 2       svc  0.843342   0.840636  0.879459  0.857877
    # 0   log_reg  0.848784   0.852438  0.874495  0.861510
    # 6   g_boost  0.859678   0.856727  0.891868  0.872547
    # 7  xg_boost  0.862408   0.857420  0.896420  0.875311
    # 5  r_forest  0.862408   0.847039  0.911016  0.876647
    # 8  lg_boost  0.880114   0.868837  0.919802  0.892273 <- BestModel

    # Stacking and Blending
    # =========================================================================================
    # Fit the models
    s_cls.fit(X_train.values, y_train.values)
    for _, estimator in estimators:
        estimator.fit(X_train, y_train)

    # Blend the models
    def blend_models_predict(x: pd.DataFrame, models: list[tuple], stacking_cls) -> pd.Series:
        prediction = pd.Series([0 for _ in range(len(x))])
        threshold = (len(models) + 1) // 2
        for _, model in models:  # Note: models and stacking_cls were already .fit()
            prediction += model.predict(x)

        prediction += stacking_cls.predict(x.values)
        return prediction.apply(lambda a: 1 if a > threshold else 0)  # if more than half of the models voted


    # Results
    # =========================================================================================
    best_model_prediction = lg_boost.predict(X_test)
    s_classifier_prediction = s_cls.predict(X_test.values)
    blend_models_prediction = blend_models_predict(X_test, estimators, s_cls)

    print('F1 score with best model:', f1_score(y_test, best_model_prediction))
    print('F1 score with stacking classifier:', f1_score(y_test, s_classifier_prediction))
    print('F1 score with blend models:', f1_score(y_test, blend_models_prediction))
    # F1 score with best model: 0.9279279279279279
    # F1 score with stacking classifier: 0.9315068493150684
    # F1 score with blend models: 0.9321266968325792

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed


class CustomRegressor:
    def __init__(self, n_jobs: int = -1, seed: int = 1):
        self.n_jobs = n_jobs
        self.seed = seed
        self._estimators = None

    def _get_model(self, x_, y_):
        if y_.name == 'SCHOOL AND OFFICE SUPPLIES':
            etr = ExtraTreesRegressor(n_estimators=500, n_jobs=self.n_jobs,
                                      random_state=self.seed)
            rfr = RandomForestRegressor(n_estimators=500, n_jobs=self.n_jobs,
                                        random_state=self.seed)
            br1 = BaggingRegressor(base_estimator=etr, n_estimators=10,
                                   n_jobs=self.n_jobs, random_state=self.seed)
            br2 = BaggingRegressor(base_estimator=rfr, n_estimators=10,
                                   n_jobs=self.n_jobs, random_state=self.seed)
            model = VotingRegressor([('ExtraTrees', br1), ('RandomForest', br2)])
        else:
            ridge = make_pipeline(RobustScaler(),
                                  Ridge(alpha=31.0, random_state=self.seed))
            svr = make_pipeline(RobustScaler(),
                                SVR(C=1.68, epsilon=0.09, gamma=0.07))

            model = VotingRegressor([('ridge', ridge), ('svr', svr)])

        model.fit(x_, y_)
        return model

    def fit(self, x_, y_):
        self._estimators = Parallel(n_jobs=self.n_jobs, verbose=0) \
            (delayed(self._get_model)(x_, y_.iloc[:, i]) for i in range(y_.shape[1]))

    def predict(self, x_):
        y_pred = Parallel(n_jobs=self.n_jobs, verbose=0) \
            (delayed(e.predict)(x_) for e in self._estimators)

        return np.stack(y_pred, axis=1)


if __name__ == '__main__':
    pass

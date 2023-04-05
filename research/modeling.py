import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


class Modeling(object):
    def performance(self, preds: [], actual: []) -> dict:
        """
        Private function for performance calculation
        """
        # get r2
        r2 = r2_score(actual, preds)
        # get rmse
        rmse = mean_squared_error(actual, preds, squared=False)
        # get mape - need to remove values when actual = 0 bc mape will div 0
        df = pd.DataFrame({'preds': preds, 'actual': actual})
        df = df[df['actual'] != 0]
        mape = mean_absolute_percentage_error(df['actual'], df['preds'])
        perf = {'r2': r2, 'rmse': rmse, 'mape': mape}
        return perf

    def lasso_regression(self, df_inp: pd.DataFrame, x_col: [], y_col: str) -> dict:
        """
        """
        df = df_inp.copy()
        # train/test split
        x_train, x_test, y_train, y_test = train_test_split(df[x_col], df[y_col], test_size=0.2,
                                                            random_state=42)
        # train model using cross validation to compute optimal alpha
        model = LassoCV(cv=5, random_state=0).fit(x_train, y_train)
        # fit model train/test
        x_train['preds_train'] = model.predict(x_train)
        x_test['preds_test'] = model.predict(x_test)
        # coefficients
        coefs = pd.DataFrame({'features': ['Intercept'] + model.feature_names_in_.tolist(),
                             'coefficients': [model.intercept_] + model.coef_.tolist()})
        # performance
        perf = {'train': self.performance(x_train['preds_train'], y_train),
                'test': self.performance(x_test['preds_test'], y_test)}
        return {'df_preds_train': x_train, 'df_preds_test': x_test, 'performance': perf,
                'model': model, 'coefficients': coefs, 'alpha': model.alpha_}

    def _importance(self, model) -> pd.DataFrame:
        """
        Private function for feature importance and standard deviation
        """
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        imp_df = pd.DataFrame({'feature': model.feature_names_in_, 'importance': importances,
                               'std': std})
        imp_df.sort_values(['importance'], ascending=False, inplace=True)
        return imp_df

    def random_forest_regression(self, df_inp: pd.DataFrame, x_col: [], y_col: str) -> dict:
        """
        """
        df = df_inp.copy()
        # initialize rf
        rf = RandomForestRegressor()
        # cross validated performance
        perf = {'r2': np.mean(cross_val_score(rf, df[x_col], df[y_col], cv=5, scoring='r2')),
                'rmse': -1*np.mean(cross_val_score(rf, df[x_col], df[y_col], cv=5,
                                                scoring='neg_root_mean_squared_error'))}
        # importance
        rf.fit(df[x_col], df[y_col])
        imp_df = self._importance(rf)
        return {'performance': perf, 'importance': imp_df}

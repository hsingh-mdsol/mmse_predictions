# Author: Hartaig Singh

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error


class ModelingTimeSeries(object):
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

    def rf_regressor(self, df_inp: pd.DataFrame, x_col: [], y_col: str, n_estimators: int = 100,
                     criterion: str = 'squared_error', max_depth: int = None) -> dict:
        """
        Fits a random forest regression model with specified hyperparameters and returns
        predictions, model object, feature importance, and performance
        """
        df = df_inp.copy()
        # fit model on full data
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                      max_depth=max_depth)
        model.fit(df[x_col], df[y_col])
        # fit model train/test split on time series split
        train_df = df.drop(df.groupby('PATIENT_KEY').tail(1).index, axis=0)
        test_df = df.groupby('PATIENT_KEY').last().reset_index()
        model_test = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                           max_depth=max_depth)
        model_test.fit(train_df[x_col], train_df[y_col])
        # predictions
        df['preds_full'] = model.predict(df[x_col])
        train_df['preds_train'] = model_test.predict(train_df[x_col])
        test_df['preds_test'] = model_test.predict(test_df[x_col])
        # importance
        imp_df = self._importance(model)
        # performance
        perf = {'full': self.performance(df['preds_full'], df[y_col]),
                'train': self.performance(train_df['preds_train'], train_df[y_col]),
                'test': self.performance(test_df['preds_test'], test_df[y_col])}
        return {'df_preds_full': df, 'df_preds_train': train_df, 'df_preds_test': test_df,
                'importance': imp_df, 'performance': perf, 'full_model': model}

    def lasso(self, df_inp: pd.DataFrame, x_col: [], y_col: str) -> dict:
        """
        Fits a random forest regression model with specified hyperparameters and returns
        predictions, model object, feature importance, and performance
        """
        df = df_inp.copy()
        # fit model on full data
        model = LassoCV(cv=5, random_state=0).fit(df[x_col], df[y_col])
        # fit model train/test split on time series split
        train_df = df.drop(df.groupby('PATIENT_KEY').tail(1).index, axis=0)
        test_df = df.groupby('PATIENT_KEY').last().reset_index()
        model_test = LassoCV(cv=5, random_state=0).fit(train_df[x_col], train_df[y_col])
        # predictions
        df['preds_full'] = model.predict(df[x_col])
        train_df['preds_train'] = model_test.predict(train_df[x_col])
        test_df['preds_test'] = model_test.predict(test_df[x_col])
        # coefficients
        coefs = pd.DataFrame({'features': ['Intercept'] + model.feature_names_in_.tolist(),
                             'coefficients': [model.intercept_] + model.coef_.tolist()})
        coefs = coefs.sort_values(['coefficients'], ascending=False)
        # performance
        perf = {'full': self.performance(df['preds_full'], df[y_col]),
                'train': self.performance(train_df['preds_train'], train_df[y_col]),
                'test': self.performance(test_df['preds_test'], test_df[y_col])}
        return {'df_preds_full': df, 'df_preds_train': train_df, 'df_preds_test': test_df,
                'importance': coefs, 'performance': perf, 'full_model': model}

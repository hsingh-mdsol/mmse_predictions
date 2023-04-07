import pandas as pd
import numpy as np


class PreprocessingTimeSeries:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        df_out['drop'] = np.where((df_out['ICD_CODE'].isna()) & (df_out['CPT_CODE'].isna()) &
                                  (df_out['DRUG_NDC'].isna()), 1, 0)
        df_out = df_out[df_out['drop'] == 0]
        df_out = df_out.sort_values(['PATIENT_KEY', 'DATE'])
        # replace none with nan
        df_out = df_out.fillna(value=np.nan)
        return df_out.drop(['drop'], axis=1)

    def forward_fill(self, df: pd.DataFrame, col: str, group: str = None) -> pd.DataFrame:
        """
        """
        df_out = df.copy()
        if group is not None:
            df_out[col] = df.groupby(group)[col].ffill()
        else:
            df_out[col] = df[col].bfill()
        return df_out

    def one_hot(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Performance one-hot encoding on specified feature column. Necessary for some tree based models
        """
        df_out = pd.concat([df, pd.get_dummies(df[[col]])], axis=1)
        return df_out

    def lag_dv(self, df: pd.DataFrame, dv: str, periods: int = 3, group: str = None) -> \
            pd.DataFrame:
        """
        """
        df_lag = df.copy()
        for i in range(1, periods + 1):
            if group is None:
                df_lag[f"{dv}_lag{i}"] = df_lag[dv].shift(i, fill_value=0)
            else:
                df_lag[f"{dv}_lag{i}"] = df_lag.groupby(group)[dv].shift(i, fill_value=0)

        return df_lag


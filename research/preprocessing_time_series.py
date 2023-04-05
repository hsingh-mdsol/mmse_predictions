import pandas as pd
import numpy as np


class PreprocessingTimeSeries:
    def clean(self, df: pd.DataFrame):
        """
        """
        df_out = df.copy()
        df_out['drop'] = np.where(df_out['VALUE'].isna() | ((df_out['ICD_CODE'].isna()) &
                                                            (df_out['CPT_CODE'].isna()) &
                                                            (df_out['DRUG_NDC'].isna())), 1, 0)
        df_out = df_out[df_out['drop'] == 0]
        df_out = df_out.sort_values(['PATIENT_KEY', 'DATE'], ascending=[False, False])
        return df_out.drop(['drop', 'GENDER_FEMALE'], axis=1)

    def make_binary(self, df: pd.DataFrame, cols: []) -> pd.DataFrame:
        """
        """
        df1 = df.drop(cols, axis=1)
        df2 = df[cols]
        df2[df2 >= 1] = 1
        df_out = pd.concat([df1, df2], axis=1)
        return df_out

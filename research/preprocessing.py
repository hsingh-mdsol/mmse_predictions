import pandas as pd


class Preprocessing:
    def long_to_wide(self, df: pd.DataFrame, index: str, columns: str, values: str, new_col_header: str):
        """
        """
        df_wide = df[[index, columns, values]].dropna().drop_duplicates()
        df_wide = pd.pivot(df_wide, index=index, columns=columns, values=values).reset_index().fillna(0)
        df_wide = df_wide.rename_axis(None, axis=1)
        df_wide.columns = [index] + [f"{new_col_header}_{x}" for x in df_wide.drop([index], axis=1).columns]
        return df_wide

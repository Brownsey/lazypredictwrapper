import pandas as pd


def recipe_001_pre_processing(
        data_df: pd.DataFrame,
        columns_to_drop: list) -> pd.DataFrame:
    """currently, this function drops columns! but can be extended to perform more steps"""

    # drop some columns
    print(data_df.columns)
    if set(columns_to_drop).issubset(data_df.columns):
        df = data_df.drop(columns_to_drop, 1)

    else:
        print(data_df.columns)
        raise KeyError(f"{columns_to_drop} not in df, can't drop it!")

    return df

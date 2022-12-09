import sys
from classes.HDataLoader import HDataLoader
import os
import pandas as pd


def recipe_000_ingest_data(
        data_path_str: str = "data/data_clean.csv",
        description_path_str: str = "data/site_descriptions.tsv"
) -> (pd.DataFrame, pd.DataFrame):
    """this recipe reads in the challenge data"""

    hdl = HDataLoader()
    data_path = os.path.abspath(data_path_str)
    description_path = os.path.abspath(description_path_str)

    data_df = hdl.load_a_text_file(location=data_path)
    print(data_df.head())

    description_df = hdl.load_a_text_file(description_path, "\t")
    print(description_df.head())

    return data_df, description_df


if __name__ == "__main__":
    recipe_000_ingest_data()

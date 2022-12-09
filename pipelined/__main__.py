from recipes.recipe_000_ingest_data import recipe_000_ingest_data
from recipes.recipe_001_pre_processing import recipe_001_pre_processing
from recipes.recipe_002_predict import recipe_002_predict
from recipes.recipe_003_generate_plan import recipe_003_generate_plan
import pandas as pd

# display nicely
pd.set_option("display.max_rows", 1000, "display.max_columns", None, "display.width", 1000, 'display.max_colwidth', None)


def launch(
        cpm: float = 2.31,                                                  # client objective
        cpc_target: float = 0.76,                                           # client objective
        data_path_str: str = "data/data_clean.csv",                         # cleaned data csv ready for modelling
        columns_to_drop: list = None,                                       # any columns you want removed
        y_var: str = "Click",                                               # name of response variable in data file
        regression_or_classification: str = "classification",               # type of modelling problem
        choose_model_manually: bool = True,                                 # prompt the user to select their model from all models
        output_predictions: bool = False,                                   # do you want to score a separate prediction file (for something like a challenge)
        coeffs_output_file_location: str = "data/output_coeffs.csv",        # where we ouput the file of coeffs
        prediction_file_location: str = "data/prediction.csv",              # where we input our predictions - if required
        prediction_ouput_file_location: str = "data/output_predictions.csv"     # where we ouput our predictions - if required
):
    """main entrypoint for running predictions
        runs through list of recipe scripts - each working with classes to perform key steps"""

    # Calculate click threshold from client objectives
    p_click = cpm/(cpc_target*1000)

    # run recipe scripts, calling class functions
    data_df, description_df = recipe_000_ingest_data(data_path_str)
    data_df = recipe_001_pre_processing(data_df, columns_to_drop)
    recipe_002_predict(
        data_df,
        y_var,
        regression_or_classification,
        choose_model_manually,
        output_predictions,
        coeffs_output_file_location,
        prediction_file_location,
        prediction_ouput_file_location
        )
    recipe_003_generate_plan(p_click=p_click)

    print("all data written to data directory")


if __name__ == "__main__":
    # runs the parameters for the data challenge when this .py is called
    launch(
        columns_to_drop=["uid", "Freq", "Freq Bucket"],
        choose_model_manually=False,
        output_predictions=True
    )




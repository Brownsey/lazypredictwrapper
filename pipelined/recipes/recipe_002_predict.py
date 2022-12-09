from classes.HLazyPredict import HLazyPredict
import pandas as pd
from sklearn import set_config
set_config(display='diagram')


def recipe_002_predict(
        data_df: pd.DataFrame,
        y_var: str = "Click",
        regression_or_classification: str = "classification",
        choose_model_manually: bool = True,
        output_predictions: bool = False,
        coeffs_output_file_location: str = "data/output_coeffs.csv",
        prediction_file_location: str = "data/prediction.csv",
        prediction_ouput_file_location: str = "data/output_predictions.csv"
        ):
    """calls HLazyPredict functions to perform inference"""

    # run lazy predict
    hlp = HLazyPredict(data_df, y_var)

    # this can be replaced with hlp.very_lazy_regressor in the case of a regression problem
    if regression_or_classification == 'classification':
        models, predictions = hlp.very_lazy_classifier()
    elif regression_or_classification == 'regression':
        models, predictions = hlp.very_lazy_regressor()
    else:
        raise ValueError("regression_or_classification set incorrectly, please review")

    # export output for ds review
    models.reset_index().to_csv("data/output_models.csv", index=False)

    # show the models on screen and allow the user to select one if they want
    print(models)
    if choose_model_manually:
        which_model = input("which model would you like?: ")
    else:
        which_model = 'LogisticRegression'

    # select the right model pipeline
    ppl = hlp.get_pipelie_object(which_model)

    # get predictions and apply them to the predictions file
    # useful for the challenge moreso than the everyday application
    if output_predictions:
        prediction_data = pd.read_csv(prediction_file_location)
        prediction_data['prediction'] = pd.DataFrame(ppl.predict(prediction_data))
        print(prediction_data.head())
        prediction_data.to_csv(prediction_ouput_file_location, index=False)

    # get marginal probabilities for each coeff from the desired pipeline
    coeffs = hlp.get_marginal_probabilities()
    print(coeffs.head())

    # write them to disk
    coeffs.to_csv(coeffs_output_file_location, index=False)


import pandas as pd


def recipe_003_generate_plan(
        coeffs_file_location: str = 'data/output_coeffs.csv',
        p_click: float = 0.003,
        output_file_location: str = 'data/output_coeffs_actions.csv'

):
    """applies thresholds to marginal probabilities to create plan"""

    # read a coeffs file from the harddisk - this is to allow this function to be called in isolation,
    # handy for running the logic without having to re-run the models.
    coeffs = pd.read_csv(coeffs_file_location)

    # apply threshold...
    coeffs['keep'] = coeffs['p'] >= p_click

    print(coeffs)
    # output
    coeffs.to_csv(output_file_location, index=False)


if __name__ == "__main__":
    recipe_003_generate_plan(
        coeffs_file_location='../data/output_coeffs.csv',
        output_file_location='../data/output_coeffs_actions.csv'
    )

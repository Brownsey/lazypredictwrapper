import pandas as pd
import pickle
import os
from scipy import stats
import sweetviz as sv
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import scipy.stats as st
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#TODO:Add doc strings to ALL functions

def get_missing_column_values(df, col = "nan",val = "nan"):
    """Generates a table containing the columns contain missing data
     and their missing data counts

    Args:
        df (pd.DataFrame): df to apply it to
    Returns:
        df (pd.DataFrame): new table containing columns with missing data and their ratio
    """

    count = df.isna().sum()

    df = (pd.concat([count.rename('missing_count'),
                       100* count.div(len(df))
                            .rename('missing_percentage')], axis = 1)
                .loc[count.ne(0)]).reset_index().rename(columns={"index":"column"}).round(2)
    
    return df.sort_values("missing_percentage", ascending=False)


def get_sweetviz(data: pd.DataFrame, y_var: str = "None", name = "sweetviz_report.html") -> None:
    cwd = os.getcwd()
    path = cwd + "\\data\\reports/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    if y_var != "None":
        my_report = sv.analyze(data, target_feat= y_var)
        my_report.show_html(path + name)
    else:
        my_report = sv.analyze(data)
        my_report.show_html(path + name)


def get_eda(data: pd.DataFrame):
    print("Total Columns:" + str(len(data.columns)))
    print("Number of Categorical Features: ", len([x for x in data.select_dtypes(include=['object'])]))
    print("Number of Numeric Features:", len([x for x in data.select_dtypes(exclude=['object'])]))
    print("Column typing:")
    print(data.dtypes)
    print("---------------------------------------------------------------------------------")
    print("Head of the Data:")
    print(data.head())
    print("Describe Summary:")
    print(data.describe())
    
    #Create sweetviz report if it doesn't already exist #TODO: add checks and if valid run sweetviz
    #if not (os.path.exists('./data/reports/sweetviz_report.html')):
    #    get_sweetviz(data, y_var)

#pickle functions for saving model outputs and option to import them back in
def pickle_model(params, file: str) -> None:
    f = open(file,"wb")
    pickle.dump(params,f)
    f.close()

def import_pickled_model(file):    
    infile = open(file,'rb')
    pickled_model = pickle.load(infile)
    infile.close()
    return pickled_model


def print_value_counts(data):
    for i in data.columns:
        print(pd.DataFrame(data[i].value_counts(dropna = False)))

def no_yes_aligner(row):
    if(row == "No"):
        return "N"
    elif(row == "Yes"):
        return "Y"
    else:
        return row

def y_aligner(row):
    if(row == "N"):
        return 0
    elif(row == "Y"):
        return 1
    else:
        return row



def get_row_wise_mode_counts(predictions_df):
    #TODO: Add 
    a = predictions_df.values.T
    b = stats.mode(a)
    predictions_df['mode'] = b[0][0]
    predictions_df['modal_count'] = b[1][0]
    return predictions_df

def ND_updator(row):
    # To handle the Non Disclosure/not happened events ~ the 825 which had ND will switch to binary
    if(row == -999997.0 or row == 999 or np.isnan(row) or row == "{ND}" or row == "{X}" or row == "{XX}"):
        return 1
    else:
        return 0

def get_top_model_predictions(predictions_df, models):
    #Pretty hacky code but just gets the predictions from the best model essentially
    return predictions_df[models.reset_index()["Model"].head(1).to_string().split(" ")[-1]]

def generate_coeffs_df(clf, X_train, X_test, y_train, y_test, model):
    provided_models = clf.provide_models(X_train, X_test, y_train, y_test)
    which_model = provided_models[model] # Shows pipeline

    if (hasattr(which_model.named_steps['classifier'],  "feature_importances_")):
        print("Using feature_importances_")
        coeffs = pd.DataFrame(which_model.named_steps['classifier'].feature_importances_.T, columns = ["coefficients"])
    elif(hasattr(which_model.named_steps['classifier'],  "coef_")):
        print("Using coef_")
        coeffs = pd.DataFrame(which_model.named_steps['classifier'].coef_.T, columns = ["coefficients"])
    else:
        print("No implemented method for this model")
        return

    # take feature names from df
    feature_names = pd.DataFrame(which_model[:-1].get_feature_names_out())
    feature_names = feature_names.rename({0: "name"}, axis=1)

    # concat together
    coeffs_df = pd.concat([feature_names, coeffs], axis=1)
    coeffs_df = coeffs_df.sort_values(by='coefficients', axis=0, ascending=True)
    return coeffs_df


def split_data(data, y_var, test_size = 0.25):
    return train_test_split(data.drop(y_var, axis = 1), data[y_var], test_size= test_size, random_state= 666, stratify=y_var)


### Modelling section
def get_scale_pos_weight(y):
    return round( (y.value_counts()[0]/y.value_counts()[1]),3)


def get_xgb_best_params(X, y, objective, weights, splits = 3, cv = 3, n_iter = 100):
    """Runs lots of XGB models and return the optimum hyper-parameters

    Args:
        X (pd.DataFrame): Dataframe contain the training data
        y (pd.Series): Series containing the POL_STATUS labels
        objective (str): str for the objective function, could be changed if using multi classification to e.g "multi:softmax"
        splits (int, optional): Number of data splits. Defaults to 3.
        cv (int, optional): Number of cross validations. Defaults to 3.
        n_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        dict: dict containing best XGB parameters
    """

    best_params = []
    best_score = []

    #Kfold Cross validation part of the code
    kf = StratifiedKFold(n_splits = splits)
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        estimator = xgb.XGBClassifier(nthreads=-1, use_label_encoder = False,
         objective= objective, tree_method = "hist", enable_categorical = True) # Can remove tree_method = hist and enable categorical to run with pd.getdummied data
        params = {  
            "n_estimators": st.randint(3, 40),
            "max_depth": st.randint(3, 40),
            "learning_rate": st.uniform(0.05, 0.4),
            "colsample_bytree": st.beta(10, 1),
            "subsample": st.beta(10, 1),
            "gamma": st.uniform(0, 5),
            'objective': [objective],
            'scale_pos_weight': [weights],
            "min_child_weight": st.expon(0, 50),
            "max_delta_step" : st.randint(1, 3)
        }

        clf = RandomizedSearchCV(estimator, params, cv = cv,
                                n_iter = n_iter, scoring="roc_auc") 
                                    
        clf.fit(X_train, y_train.values)  

        best_params.append(clf.best_params_)
        best_score.append(clf.best_score_)   

    #Getting best parameters from CV
    best_params = best_params[best_score.index(max(best_score))]
    #potentially grab mean/std dev here
    return best_params, best_score.index(max(best_score))

def run_best_params_xgb(best_params, X ,y):
    """Re-Runs the optimal model found from the CV run

    Args:
        best_params (dict): dict of best params

    Returns:
        clf model: returns the fitted clf XGBClassifier 
    """
    clf = xgb.XGBClassifier(nthreads=-1, **best_params, use_label_encoder= False
    , enable_categorical = True, tree_method = "hist")
    clf.fit(X, y)  
    return clf 


def print_F1_score(true, predicted):
    #prints the F1 score to console
    print("F1 Score is: " + str(round(f1_score(true, predicted, average='weighted'), 3) ) )

def print_recall_score(true, predicted):
    print("Recall Score is: " + str(round(recall_score(true, predicted), 3)))

def print_precision_score(true, predicted):
    print("Precision Score is: " + str(round(precision_score(true, predicted),3)))

def get_top_features(X, clf):
    #Gets the feature importances
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X.columns, clf.feature_importances_):
        feats[feature] = importance

    return pd.Series(feats).sort_values(ascending = False)


#pickle functions for saving model outputs and option to import them back in
def pickle_model(params, file):
    f = open(file,"wb")
    pickle.dump(params,f)
    f.close()

def import_pickled_model(file):    
    infile = open(file,'rb')
    pickled_model = pickle.load(infile)
    infile.close()
    return pickled_model


def plot_confusion_matrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()


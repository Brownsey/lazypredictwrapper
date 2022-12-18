from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import utils
import pandas as pd

"""
Modelling will be split into two sections:

The lazypredict section will be used to quickly get a feel for the data and the models that are available
It mimics a local version of cloud automl packages such as google's automl offering in Vertex AI
The main difference is that there is no hyperparameter tuning, this means the decision tree models usually perform well
but do vary with which one ranks number one, but it is very good at giving a quick overview of the data and the best models
for the data, it is built using the sklearn pipelines so all the usual pipeline information is available

The second section will be a more traditional approach with a custom XGB model.
This model was chosen as it ranked well for the lazypredict section and is a very good model in general for this sort of problem
It works by storing the best parameters from the the cross validated runs and using the best parameters for the final model which is
then used to make predictions on the test set

"""
modelling_data = pd.read_csv("data/modelling_data_clean.csv")
X = modelling_data.drop(columns = "Sale")
y = modelling_data.Sale

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25,random_state = 666)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models.to_csv("data/models.csv")
print(models)

#This doesn't work well if there is inbalanced datasets as less good models tend to predict solely majority class
modal_predictions = utils.get_row_wise_mode_counts(predictions)

#Get's the predictions of the best model
best_predictions = utils.get_top_model_predictions(predictions, models)
best_predictions.to_csv("data/best_predictions.csv", index = False)

#This returns the coeffiecents of a given model, haven't updated it to work with tree models yet

utils.generate_coeffs_df(clf, X_train, X_test, y_train, y_test, "LGBMClassifier").to_csv("data/coeffs/lgbm_coeffs.csv")
utils.generate_coeffs_df(clf, X_train, X_test, y_train, y_test, "DecisionTreeClassifier").to_csv("data/coeffs/dt_coeffs.csv")
utils.generate_coeffs_df(clf, X_train, X_test, y_train, y_test, "RandomForestClassifier").to_csv("data/coeffs/rf_coeffs.csv")
utils.generate_coeffs_df(clf, X_train, X_test, y_train, y_test, "XGBClassifier").to_csv("data/coeffs/xgb_coeffs.csv")
#Printing for reference
print(utils.generate_coeffs_df(clf, X_train, X_test, y_train, y_test, "XGBClassifier"))


# Old school version
#Setting to Categorical as when not pipelined this is not automatically done whereas this would be handled as part of lazypredict
modelling_data[["Channel", "Product", "Smoker", "Joint?"]] = modelling_data[["Channel", "Product", "Smoker", "Joint?"]].astype("category")

#Re-running the training test split section as it is required for XGB to work
X = modelling_data.drop(columns = "Sale")
y = modelling_data.Sale

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25,random_state = 666)

##XGB method
weight = utils.get_scale_pos_weight(y)
best_params, best_score = utils.get_xgb_best_params(X_train, y_train, "binary:logistic", weights = weight)
print(best_params)
#So run can be saved and re-imported in future if required
utils.pickle_model(best_params, "data/pickles/best_params.pkl")
best_params = utils.import_pickled_model("data/pickles/best_params_final.pkl")#Just shows how to import a pickled model of a model
clf = utils.run_best_params_xgb(best_params, X_train , y_train)

clf_predict = clf.predict(X_test)
pd.DataFrame(clf_predict).to_csv("data/clf_predict.csv", index = False)

"""
#If you wanted to implement any _proba methodology would be done as follows
clf_predict_proba = (clf.predict_proba(X_test)[:,1] >= 0.5).astype(int)
pd.DataFrame(clf_predict_proba).to_csv("data/coeffs/clf_predict_proba.csv", index = False)
utils.plot_confusion_matrix(y_test, clf_predict_proba)
"""

#Get the top features from the model
top_features = utils.get_top_features(X, clf)
print(top_features)
top_features.to_csv("data/coeffs/xgb_top_features.csv", index = False)


utils.print_F1_score(y_test, clf_predict)

utils.plot_confusion_matrix(y_test, clf_predict)

#TODO: Would be interesting to look at the extra columns that are dropped because of nulls and see if they are important

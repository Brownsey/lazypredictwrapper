from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np


class HLazyPredict:

    def __init__(self, df: pd.DataFrame, y_var: str, random_state: int = 666,
    modelling_type:str = "classifier", test_size: float = 0.2):
        self.df = df                                    # input df
        self.y_var = y_var                              # input y var colname
        self.random_state = random_state
        self.test_size = test_size                      # input proportion of test:train
        self.x, self.y = self.__split_df_into_x_y()     # dfs of predictors and response variables, respectively
        self.x_train, self.x_test, self.y_train, self.y_test = self.__get_test_train_split(self.x, self.y)  # x and y subdivided into train and test
        self.has_modelled = False                       # boolean flag for catching implementation errors
        self.__vlp = None                               # output of very lazy modelling run
        self.__provided_models = dict()                 # dictionary of all pipelines, keyed from model name
        self.__models = pd.DataFrame()                  # df of model performance
        self.__predictions = pd.DataFrame()             # predictions of for each row, by model attempted by vlp
        self.__pipeline = None                          # winning pipeline object selected from lazy predict
        self.model_type = None                          # string containing which model was selected
        self.__marginal_df = pd.DataFrame()             # used as a nasty hack for calculating marginal probabilities
        self.modelling_type = modelling_type

    def __split_df_into_x_y(self) -> (pd.DataFrame, pd.Series):
        """takes a df and breaks it out into two parts, x and y"""

        if self.y_var in self.df:
            y = self.df[self.y_var]
            x = self.df.drop(self.y_var, 1)
            return x, y
        else:
            raise ValueError(f"{self.y_var} not found in df for lazy prediction...")

    def __get_test_train_split(self, x: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
        """splits out x and y parts into train and test"""

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        return x_train, x_test, y_train, y_test

    def very_lazy_classifier(self) -> (pd.DataFrame, pd.DataFrame):
        """runs the lazy classifier, also stores model files into self.provided_models"""

        self.__vlp = LazyClassifier(verbose=0, predictions=True)
        models, self.__predictions = self.__vlp.fit(self.x_train, self.x_test, self.y_train, self.y_test)
        self.__models = models.reset_index()
        self.has_modelled = True
        self.__get_provided_models()
        return self.__models, self.__predictions

    def very_lazy_regressor(self) -> (pd.DataFrame, pd.DataFrame):
        """runs the lazy regressor, also stores model files into self.provided_models"""

        self.__vlp = LazyRegressor(predictions=True)
        self.__models, self.__predictions = self.__vlp.fit(self.x_train, self.x_test, self.y_train, self.y_test)
        self.has_modelled = True
        self.__get_provided_models()
        return self.__models, self.__predictions

    def get_models(self) -> pd.DataFrame:
        """returns models after modelling has been done"""

        if self.has_modelled:
            return self.__models
        else:
            raise ValueError("modelling hasn't been done yet! run a model first")

    def get_predictions(self) -> pd.DataFrame:
        """returns predictions after modelling has been done"""

        if self.has_modelled:
            return self.__predictions
        else:
            raise ValueError("modelling hasn't been done yet! run a model first")

    def __get_provided_models(self) -> dict:
        """once models have been run, this pulls a dictionary of pipelines, keyed off the model names"""

        if self.has_modelled:
            self.__provided_models = self.__vlp.provide_models(self.x_train, self.x_test, self.y_train, self.y_test)
            return self.__provided_models

    def get_pipeline_object(self, which_model: str) -> pipeline:
        """returns the pipeline object you just selcted, also saves it in self.__pipeline"""

        if self.has_modelled and which_model in self.__provided_models:
            self.__pipeline = self.__provided_models[which_model]
            self.model_type = which_model
            return self.__pipeline

    def get_pipeline_coeffs(self) -> pd.DataFrame:
        """where appropriate, pull coeffs from a pipeline"""

        if self.__pipeline:

            # pull coeff information from pipeline
            coeffs = self.__pipeline.named_steps['classifier'].coef_.tolist()
            coeffs = pd.DataFrame(coeffs).transpose()
            coeffs = coeffs.rename({0: "coeff"}, axis=1)

            # take feature names from df
            feature_names = pd.DataFrame(self.x.columns)
            feature_names = feature_names.rename({0: "name"}, axis=1)

            # concat together
            coeffs_df = pd.concat([feature_names, coeffs], axis=1)
            coeffs_df = coeffs_df.sort_values(by='coeff', axis=0, ascending=True)
            print(coeffs_df.head())

            return coeffs_df

    def get_marginal_probabilities(self):
        """extensible function for getting probabilities from various pipelines
            LogisticRegression is provided here for POC"""

        self.__generate_marginal_df()
        coeffs = self.get_pipeline_coeffs()
        coeffs['p'] = 0.0000

        # todo: break this section out into a separate function
        # gets correct probability prediction function based on model type
        if self.model_type == "LogisticRegression":
            self.__marginal_df = self.__get_logistic_probabilities(self.__marginal_df)

        # example of extending this function
        # elif self.model_type == "LinearSVC":
        #     self.__marginal_df = self.__get_linearsvc_probabilities()

        else:
            raise ValueError("model type not currently supported for marginal probabilities, consider extending this function")

        for c in range(0, self.x.columns.size):
            big_prob = self.__marginal_df.at[(c*2)+1, "prob"]
            little_prob = self.__marginal_df.at[c*2, "prob"]
            print(f"{self.x.columns[c]}, {big_prob} - {little_prob} = {big_prob - little_prob}")

            coeffs.at[c, "p"] = big_prob - little_prob 

        coeffs = coeffs.sort_values(["p"], ascending=True)

        return coeffs

    def __get_linearsvc_probabilities(self):
        """gets probabilities from linearSVC models using platt scaling"""
        # todo rework this with more time
        # if self.model_type == "LinearSVC":
        #     clf = CalibratedClassifierCV(self.__pipeline)
        #     clf.fit(self.x_train, self.y_train)
        #     y_proba = clf.predict_proba(self.x_test)
        #     print(y_proba)
        #     return y_proba
        # else:
        #     raise TypeError("model type is not LinearSVC")
        pass

    def __get_logistic_probabilities(self, df: pd.DataFrame):
        """returns probabilities for marginal df"""
        if self.model_type == "LogisticRegression":

            df['prob'] = self.__pipeline.predict_proba(df)[:, 1]

            return df

    def __generate_marginal_df(self):
        """makes a dataframe of 1 and 2 observations for each column"""
        number_of_columns = self.x.columns.size
        # make a df full of zeroes
        marginal_df = pd.DataFrame(np.zeros((number_of_columns*2, number_of_columns)))

        # start on row 0
        r = 0

        # nasty nested for loop
        for c in range(0, number_of_columns):
            for v in (30, 31):
                marginal_df.iloc[r, c] = v
                r += 1
            marginal_df = marginal_df.rename({c: self.x.columns[c]}, axis=1)

        # store your dirty deed
        self.__marginal_df = marginal_df

        return marginal_df

    
    def get_row_wise_mode_counts(self, top_x):
    #TODO: Add code to only select top X predictions
    #TODO: Think about best way to select top X - should it be off a threshold accuracy?
    # or within X % of the top accuracy?
    # Essemtially would act as a bagging approach for the top X models chosen
    #Current downsides is it uses all models - some of which are very bad and the combination is not better than the best model
        predictions = self.__predictions
        p = self.__predictions.values.T
        m = stats.mode(p)
        predictions['mode'] = m[0][0]
        predictions['modal_count'] = m[1][0]
        return predictions



    def __get_top_model(self, position  = 1):
        #Return model name at position in the list of models
        models = self.__models
        # This is needed because the index is included in the model name like '0    NearestCentroid'
        return models["Model"].head(position).to_string().split(" ")[-1]

    def get_model_predictions(self, model):
        #Pretty hacky code but just gets the predictions from the best model essentially 
        return self.__predictions[model] 


    def generate_coeffs_df(self, model):
        
        """
        #To have a play with this, you can use the following code:
        # This would use the cognizant data
        X = modelling_data.drop(columns = "Churn")
        y = modelling_data.Churn


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25,random_state = 666)

        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, predictions=True)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        #models.to_csv("data/models.csv")
        print(models)

        provided_models = clf.provide_models(X_train, X_test, y_train, y_test)
        which_model = provided_models["Perceptron"] # Shows pipeline
        which_model.named_steps['classifier']
        dir(which_model.named_steps['classifier']) #Shows methods associated with each pipelined model
        """

        which_model = self.get_pipeline_object(model)

        if (hasattr(which_model.named_steps['classifier'],  "feature_importances_")):
            print("Using feature_importances_")
            coeffs = pd.DataFrame(which_model.named_steps['classifier'].feature_importances_.T, columns = ["coefficients"])
        elif(hasattr(which_model.named_steps['classifier'],  "coef_")):
            print("Using coef_")
            coeffs = pd.DataFrame(which_model.named_steps['classifier'].coef_.T, columns = ["coefficients"])
        elif(hasattr(which_model.named_steps['classifier'], "centroids_")):
            print("Using centroids_")
            print("This method returns the centroids of the clusters, not the coefficients")
            print("To understand feature importances force this to run on a model with coef_ or feature_importances_")
            return pd.DataFrame(which_model.named_steps['classifier'].centroids_) # This doesnt
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

    def get_top_x_models(self, accuracy_metric = "Balanced Accuracy", margin = 0.05 ):
        #Subsets the models df down to just be the top X models that are within a margin of the top model for a given accuracy metric
        models = self.__models
        top_accuracy = float(models.head(1)[accuracy_metric])
        models = models[models["Balanced Accuracy"] > top_accuracy - margin]
        self.__top_x_models = models
        self.num_top_x = len(models.index)


    def run_modelling(self):
        """main function for testing"""
        if self.modelling_type == "classifier":
            self.very_lazy_classifier()
        elif self.modelling_type == "regressor":
            self.very_lazy_regressor()
        else:
            raise ValueError("modelling type not supported, please input either classifier or regressor")
        
        #TODO: Update config to decide which functions get run and 
        
        models = self.get_models()
        predictions = self.get_predictions()
        #get predictions of just the top model
        model_name = self.__get_top_model()
        print("best model is: " + model_name)
        top_predictions = self.get_model_predictions(model = model_name)
        #TODO: Update the model part to get coefficients for top X models rather than just top 1
        coeffs_df = self.generate_coeffs_df(model= model_name)
        #pipeline_coeffs = self.get_pipeline_coeffs() # This sometimes fails as the coeffs are not defined for all models

        return models, predictions, top_predictions, coeffs_df
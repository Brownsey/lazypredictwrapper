from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import pickle



class HLazyPredict:

    def __init__(self, df: pd.DataFrame, y_var: str, random_state: int = 666,
    modelling_type:str = "classifier", test_size: float = 0.2, margin = 0.05):
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
        self.margin = margin

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

   
    def get_row_wise_mode_counts(self):
    #TODO: Add code to only select top X predictions
    #TODO: Think about best way to select top X - should it be off a threshold accuracy?
    # or within X % of the top accuracy?
    # Essentially would act as a bagging approach for the top X models chosen
    # Current downsides is it uses all models - some of which are very bad and the combination is not better than the best model
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


    def generate_coeffs_df(self, model, position = 1):
        
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
            print("Analysing: " + model +" Using feature_importances_" + " at list postion " + str(position -1))
            coeffs = pd.DataFrame(which_model.named_steps['classifier'].feature_importances_.T, columns = ["coefficients"])
        elif(hasattr(which_model.named_steps['classifier'],  "coef_")):
            print("Analysing: " + model +" Using coef_" + " at list postion" + str(position -1))
            coeffs = pd.DataFrame(which_model.named_steps['classifier'].coef_.T, columns = ["coefficients"])
        elif(hasattr(which_model.named_steps['classifier'], "centroids_")):
            print("Analysing: " + model + " Using centroids_" + " at list postion " + str(position -1))
            print("This method returns the centroids of the clusters, not the coefficients")
            print("To understand feature importances force this to run on a model with coef_ or feature_importances_")
            return pd.DataFrame(which_model.named_steps['classifier'].centroids_) # This doesnt
        elif(hasattr(which_model.named_steps['classifier'], "feature_log_prob_")):
            #This one uses the feature log probabilities for both the positive and negative classes
            #Not currently implemented if > 2 classes
            print("Analysing: " + model + " Using feature_log_prob_" + " at list postion " + str(position -1))
            coeffs = pd.DataFrame(which_model.named_steps['classifier'].feature_log_prob_[0: ].T, columns = ["negative_coefficients", "coefficients"])
        
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


    #Currently only works for binary classification
    def generate_proba_predictions(self, model, data = None, position = 1, threshold = 0.5):
            
            #TODO: Expand to more than binary classification

            which_model = self.get_pipeline_object(model)
            if data is None:
                data = self.X_test

            if (hasattr(which_model.named_steps['classifier'],  "predict_proba")):
                print("Proba analysing: " + model +" predict_proba" + " at list postion " + str(position -1))
                return (which_model.predict_proba(data)[:,1] > threshold).astype(int)
            else:
                print("No implemented method for " + model)
                return


    def save_pipeline_object(self, model, file = "data/best_model_pipeline.pkl") -> None:
        #Save the pipeline object for a given model
        params = self.get_pipeline_object(model)
        f = open(file,"wb")
        pickle.dump(params,f)
        f.close()

    def get_top_x_models(self, accuracy_metric = "Balanced Accuracy"):
        #Subsets the models df down to just be the top X models that are within a margin of the top model for a given accuracy metric
        models = self.__models
        margin = self.margin
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
        self.get_top_x_models()
        num_top_x = self.num_top_x

        top_predictions = self.get_model_predictions(model = model_name)
        self.top_predictions = top_predictions
        #TODO: Update the model part to get coefficients for top X models rather than just top 1

        coeff_list = list()
        for i in range(1, num_top_x + 1):
            coeffs_df = self.generate_coeffs_df(model= self.__get_top_model(position = i), position = i)
            coeff_list.append(coeffs_df)
        
        # save the best model as a pickle file that can be imported later for deployment/usage
        self.save_pipeline_object(model_name)





        #pipeline_coeffs = self.get_pipeline_coeffs() # This sometimes fails as the coeffs are not defined for all models

        return models, predictions, top_predictions, coeff_list



 # =========================== Visualisations/Model Performance Code ===========================
    def plot_confusion_matrix(self):
        cwd = os.getcwd()
        path = cwd + "/data/reports/" + "confusion_matrix.png"

        y = self.y_test
        #Must be run after modelling
        preds = self.top_predictions
        # calculate confusion matrix
        matrix = confusion_matrix(y, preds)

        # plot confusion matrix
        _, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(x=j, y=i, s=matrix[i, j], va="center", ha="center")

        # axis labels
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # save the plot
        plt.savefig(path)
        "Print saved the confusion matrix to: " + path
        plt.close()


    def calculate_FPR_TPR(self):
        y = self.y_test
        preds = self.top_predictions
        FPR = sum((preds == 1) & (y == 0)) / sum(y == 0)
        TPR = sum((preds == 1) & (y == 1)) / sum(y == 1)
        return FPR, TPR

    def __calculate_gmean(self):

        FPR, TPR = self.calculate_FPR_TPR()
        return np.sqrt(TPR * (1 - FPR))
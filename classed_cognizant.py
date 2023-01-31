from classes.HEDA import HEDA
from classes.HDataLoader import HDataLoader
from classes.HPreProcess import HPreProcess
from classes.HLazyPredict import HLazyPredict
from utils.utils import *
import os
import pandas as pd

#Required Paths
eda_config_location = "data/cognizant_config.json"
data_path_str = "data/AIA_Churn_Modelling_Case_Study.csv"
pre_process_config_location = "data/cognizant_preprocess_config.json"

#Initialise the data loader
hdl = HDataLoader()

#Loading the data
eda_config = hdl.load_a_json(location=os.path.abspath(eda_config_location))
data = hdl.load_a_csv(location=os.path.abspath(data_path_str))
preprocess_config = hdl.load_a_json(location= os.path.abspath(pre_process_config_location))

#Eda section
print("Running the EDA class")
#eda = HEDA(data, config=eda_config)
#eda.run_eda()

##### Pre-processing section
preprocess = HPreProcess(data, config = preprocess_config)
#preprocess.column_dropper()
data = preprocess.run_preprocess()

### Modelling section

hlp = HLazyPredict(data, y_var= "Churn", margin = 0.1)

models, predictions, top_predictions, coeffs_df = hlp.run_modelling()
hlp.plot_confusion_matrix()

models
coeffs_df[4]

#Data saving section
models.to_csv("data/cognizant_models.csv")
top_predictions.to_csv("data/cognizant_top_predictions.csv")
coeffs_df[4].to_csv("data/cognizant_coeffs_df.csv")


#Model deployment idea
"""
When running the model class, we can save the best model pipeline object as a pickle file
This can then be imported and used to make predictions on new data
Then over time, we can re-run the lazypredict methodology which will
retrain all the models and update the pickle file
"""
test_model = import_pickled_model("data/best_model_pipeline.pkl")
x_test = data.sample(n=1).drop(columns = "Churn")
if (hasattr(test_model,  "predict_proba")):
       print("Using predict_proba")
       prediction =  (test_model.predict_proba(x_test)[:,1] > 0.5).astype(int)[0]
elif (hasattr(test_model,  "predict")):
       print("Using predict")
       prediction = (test_model.predict(x_test)).astype(int)[0]
else:
       prediction = "No Prediction method implemented for this model"
print("The prediction is" + str(prediction))

#This can then be deployed via a fastapi or flask app and deployed on cloudrun and then accessed via the requests library as an endpoint
#Cloud run - Good scalability based on demand so in GCP a good option
#Other options - IN GCP would be building the entire pipeline in the Vertex AI platform
       
       
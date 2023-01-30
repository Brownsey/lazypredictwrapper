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


#Model deployment idea - need to ensure that the model that is saved has the predict_proba method
test_model = import_pickled_model("data/best_model_pipeline.pkl")
x_test = data.sample(n=1).drop(columns = "Churn")
(test_model.predict_proba(X_test)[:,1] > 0.5).astype(int)[0]
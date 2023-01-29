from classes.eda_class import HEDA
from classes.HDataLoader import HDataLoader
from classes.HPreProcess import HPreProcess
from classes.HLazyPredict import HLazyPredict
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


print("Running the EDA class")
#eda = HEDA(data, config=eda_config)
#eda.run_eda()

##### Pre-processing example code:
preprocess = HPreProcess(data, config = preprocess_config)
#preprocess.column_dropper()
data = preprocess.run_preprocess()

### Modelling section

hlp = HLazyPredict(data, y_var= "Churn")

models, predictions, top_predictions, coeffs_df = hlp.run_modelling()


###archaic shit
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
modelling_data = data.copy()

X = modelling_data.drop(columns = "Churn")
y = modelling_data.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25,random_state = 666)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)



modelling_data1 = modelling_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
         'Churn']]   ##Works


['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn']



modelling_data1 = modelling_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', "MonthlyCharges",
         'Churn']]


X = modelling_data1.drop(columns = "Churn")
y = modelling_data1.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25,random_state = 666)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models

from classes.eda_class import HEDA
from classes.HDataLoader import HDataLoader
from classes.preprocess_class import HPreProcess
import os
import pandas as pd

data_path_str = "data/eda_config.json"
hdl = HDataLoader()
data_path = os.path.abspath(data_path_str)
eda_config = hdl.load_a_json(location=data_path)

data = pd.read_excel("data/MockDataSet1.xlsx").drop(columns= "Unnamed: 0", errors= "ignore")

"""

#Interestingly this doesn't load the xlsx, v.confusing

data_path_str = "data/MockDataSet1.xlsx"
hdl = HDataLoader()
data_path = os.path.abspath(data_path_str)
data_df = hdl.load_a_xlsx(location=data_path_str)
print(data_df.head())

"""
#Running the eda without a config file

eda = HEDA(data, y_var= "Sale", ID_col="Unnamed: 0")

eda.get_missing_column_values()
eda.get_sweetviz()
#Running the eda with a config file
print("configged eda version")
eda = HEDA(data, config=eda_config)
eda.run_eda()


##### Pre-processing example code:
hdl = HDataLoader()
data_path_str = "data/preprocess_config.json"
data_path = os.path.abspath(data_path_str)
preprocess_config = hdl.load_a_json(location=data_path)
pre_processed_data = HPreProcess(data, config = preprocess_config)
data = pre_processed_data.run_preprocess()
data.Smoker.value_counts()



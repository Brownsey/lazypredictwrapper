from eda_class import HEDA
from HDataLoader import HDataLoader
import os
import pandas as pd

data_path_str = "eda_config.json"
hdl = HDataLoader()
data_path = os.path.abspath(data_path_str)
eda_config = hdl.load_a_json(location=data_path_str)

data = pd.read_excel("data/MockDataSet1.xlsx").drop(columns= "Unnamed: 0", errors= "ignore")
#Checking Quotes

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

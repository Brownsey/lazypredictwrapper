# This file will generate an EDA class which will read in a dataset and provide insights on this dataset
import pandas as pd


class HPreProcess:

    def __init__(self, df: pd.DataFrame, config: dict = None,
     columns_to_drop: list = [], no_yes_columns = None, date_formatter_columns = None):
        self.df = df                                  # input df
        self.config = config                          # config file for running the pre-processing automatically
        self.columns_to_drop = columns_to_drop
        self.no_yes_columns = no_yes_columns
        self.data_formatter_columns = date_formatter_columns

        #Overrites if config is passed in and contains them
        if config != None:
            if "columns_to_drop" in config:
                if config["columns_to_drop"] != "None":
                    self.columns_to_drop = config["columns_to_drop"]
            if "no_yes_columns" in config:
                if config["no_yes_columns"] != "None":
                    self.no_yes_columns = config["no_yes_columns"]
            if "date_formatter_columns" in config:
                if config["date_formatter_columns"] != "None":
                    self.sweetviz_name = config["date_formatter_columns"]

    def column_dropper(self):
        self.df = self.df.drop(columns = self.columns_to_drop, errors = "ignore")
        return self.df
    
    def __no_yes_aligner(self, row):
        if(row == "No"):
            return "N"
        elif(row == "Yes"):
            return "Y"
        else:
            return row

    def run_no_yes_aligner(self):
        print("Running no_yes_aligner")
        columns = self.no_yes_columns
        self.df[columns] = self.df[columns].applymap(self.__no_yes_aligner)

    # config code to run all the column aligning code
    def run_row_aligner(self):
        if "no_yes_columns" in self.config:
            if self.config["no_yes_columns"] != "None":
                self.run_no_yes_aligner()

    def __date_feature_engineerer(self, column):
        """
        Function to do some date feature engineerig (untested)
        TODO: Test it actually works lol

        Args:
            column (str): Column to apply to feature engineering to

        Returns:
            pd.DataFrame: Returns the dataframe with original column dropped and extra columns added
        """
        data = self.df
        column1 = column + "_day_of_week"
        column2 = column + "_month"
        column3 = column + "_hour"
        column4 = column + "_minute"
        data[column1] = data[column].dt.day_name()
        data[column2] = data[column].dt.month_name()
        data[column3] = data[column].dt.hour
        data[column4] = data[column].dt.minute
        data.drop(column, axis = 1, inplace=True)
        self.data = data
        return data
    
    def run_date_feature_engineerer(self):
        columns = self.data_formatter_columns
        for i in range(0, len(columns)):
            self.__date_feature_engineerer(column = columns[i])
        return self.data

    def run_preprocess(self):
        config = self.config
        if self.config == None:
            print("Config not set! Please set config and try again")
            return

        if "drop_columns" in config:
            if config["drop_columns"] == "True":
                self.column_dropper()

        if "row_aligner" in config:
            if config["row_aligner"] == "True":
                self.run_row_aligner()

        if "date_formatter" in config:
            if config["date_formatter"] == "True":
                if self.data_formatter_columns != None:
                    self.run_date_feature_engineerer()

        return self.df

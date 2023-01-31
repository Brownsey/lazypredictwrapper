# This file will generate an EDA class which will readdata in a dataset and provide insights on this dataset
import pandas as pd


class HPreProcess:

    def __init__(self, data: pd.DataFrame, config: dict = None, binary_aligner_columns = None,
     columns_to_drop: list = [], no_yes_columns = None, date_formatter_columns = None):
        self.data = data                                  # input data
        self.config = config                          # config file for running the pre-processing automatically
        self.columns_to_drop = columns_to_drop
        self.no_yes_columns = no_yes_columns
        self.data_formatter_columns = date_formatter_columns
        self.binary_aligner_columns = binary_aligner_columns
        self.drop_duplicates = False
        self.binary_aligner = False
        self.change_type = False
        self.numeric = False
        self.object = False
        self.str = False
        

        #Overrites if config is passed in and contains them and won't break if invalid config put in
        #TODO: could probably be cleaned up a bit
        if config != None:
            if "drop_duplicates" in config:
                if config["drop_duplicates"] != "None":
                    self.drop_duplicates = True
            if "column_dropper" in config:
                if config["columns_to_drop"] != "None":
                    self.columns_to_drop = config["columns_to_drop"]
            if "no_yes_columns" in config:
                if config["no_yes_columns"] != "None":
                    self.no_yes_columns = config["no_yes_columns"]
            if "date_formatter_columns" in config:
                if config["date_formatter_columns"] != "None":
                    self.date_formatter_columns = config["date_formatter_columns"]
            if "binary_aligner" in config:
                if config["binary_aligner_columns"] != "None":
                    self.binary_aligner_columns = config["binary_aligner_columns"]
            if "change_type" in config:
                if config["change_type"] == "True":
                    self.change_type = True
                    if config["numeric"][0] != "N/A":
                        self.numeric = config["numeric"]
                    if config["object"][0] != "N/A":
                        self.object = config["object"]
                    if config["str"][0] != "N/A":
                        self.str = config["str"]
                else:
                    self.change_type = False # not required but heyho

    def column_dropper(self):
        print("dropping columns" + str(self.columns_to_drop))
        self.data = self.data.drop(columns = self.columns_to_drop, errors = "ignore")
        return self.data
    
    def __no_yes_aligner(self, row):
        if(row == "No"):
            return "N"
        elif(row == "Yes"):
            return "Y"
        else:
            return row

    def __binary_aligner(self, row):
        txt = row.lower()
        if(txt == "n" or txt == "no"):
            return 0
        elif(txt == "y" or txt == "yes"):
            return 1
        else:
            return 2 # This is an invalid value and can be used to check for errors

    def run_binary_aligner(self):
        print("Running binary_aligner")
        columns = self.binary_aligner_columns
        self.data[columns] = self.data[columns].applymap(self.__binary_aligner)

    def run_no_yes_aligner(self):
        print("Running no_yes_aligner")
        columns = self.no_yes_columns
        self.data[columns] = self.data[columns].applymap(self.__no_yes_aligner)

    # config code to run all the column aligning code
    def run_row_aligner(self):
        if "no_yes_columns" in self.config:
            if self.config["no_yes_columns"] != "None":
                self.run_no_yes_aligner()

    def update_types(self):
        #Update data types
        if self.numeric != False:
            print("Updating float columns" + str(self.numeric))
            self.data[self.numeric] = self.data[self.numeric].apply(pd.to_numeric, errors='coerce')
        if self.object != False:
            print("Updating object columns" + str(self.object))
            self.data[self.object] = self.data[self.object].astype(object)
        if self.str != False:  
            print("Updating str columns" + str(self.str))
            self.data[self.str] = self.data[self.str].astype(str)
        return self.data
        
    def __date_feature_engineerer(self, column):
        """
        Function to do some date feature engineerig (untested)
        TODO: Test it actually works lol

        Args:
            column (str): Column to apply to feature engineering to

        Returns:
            pd.DataFrame: Returns the dataframe with original column dropped and extra columns added
        """
        data = self.data
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

    #TODO: Add code to handle N/As with the following options:
    """
    Mode, Mean, Median, ML model to predict the value
    """

    def run_preprocess(self):
        config = self.config
        if self.config == None:
            print("Config not set! Please set config and try again")
            return

#This needs to be step one since if we drop columns before this, we will lose the duplicates
        if self.drop_duplicates == True: 
            self.data = self.data.drop_duplicates()                  

        if "column_dropper" in config:
            if config["column_dropper"] == "True":
                self.column_dropper()

        if "row_aligner" in config:
            if config["row_aligner"] == "True":
                self.run_row_aligner()

        if self.change_type == True:
            self.update_types()

        if "binary_aligner" in config:  
            if config["binary_aligner"] == "True":
                if self.binary_aligner_columns != None:
                    self.run_binary_aligner()

        if "date_formatter" in config:
            if config["date_formatter"] == "True":
                if self.data_formatter_columns != None:
                    self.run_date_feature_engineerer()
        
        
        return self.data

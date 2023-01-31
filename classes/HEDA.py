# This file will generate an EDA class which will read in a dataset and provide insights on this dataset
import pandas as pd
import os
import sweetviz as sv
import pandas_profiling as pp

# Print statement is for when running everything together it prints all the EDA to the window
# Return statement is for when running individuals since df returns are nicer to look at than print statements
class HEDA:

    def __init__(self, df: pd.DataFrame, y_var: str = None, ID_col = None, config = None,
     sweetviz_name = "sweetviz_report.html"):
        self.df = df                                  # input df
        self.config = config                          # config file for running EDA automatically
        
        if config != None:
            if "y_var" in config:
                if config["y_var"] != "None":
                    self.y_var = config["y_var"]
            if "ID_col" in config:
                if config["ID_col"] != "None":
                    self.ID_col = config["ID_col"]
            if "sweetviz_name" in config:
                if config["sweetviz_name"] != "None":
                    self.sweetviz_name = config["sweetviz_name"]
                else:
                    self.sweetviz_name = sweetviz_name
        else:
            self.y_var = y_var                         # Input y var colname
            self.ID_col = ID_col                       # ID column for dropping for sweetviz
            self.sweetviz_name = sweetviz_name         # Sweetviz name if required

    def get_missing_column_values(self):

        """Generates a table containing the columns contain missing data
        and their missing data counts

        Args:
            df (pd.DataFrame): df to apply it to
        Returns:
            df (pd.DataFrame): new table containing columns with missing data and their ratio
        """
        df = self.df
        count = df.isna().sum()

        df = (pd.concat([count.rename('missing_count'),
                        100* count.div(len(df))
                                .rename('missing_percentage')], axis = 1)
                    .loc[count.ne(0)]).reset_index().rename(columns={"index":"column"}).round(2)
        
        print(df.sort_values("missing_percentage", ascending=False) )
        return df.sort_values("missing_percentage", ascending=False) 

    def get_sweetviz(self) -> None:
        """
        Creates the sweetviz report for solely a comparison of dataframe
        :TODO Could be extended to included train/test splits to see further info

        Args:
            data (pd.DataFrame): dataframe
            y_var (str, optional): . Defaults to "None".
            name (str, optional): name of the report to be made. Defaults to "sweetviz_report.html".
        """
        df = self.df
        if self.ID_col != "None":
            df = df.drop(columns = self.ID_col, errors = "ignore")                 
        df_types = pd.DataFrame(df.apply(pd.api.types.infer_dtype, axis=0)).reset_index().rename(columns={'index': 'column', 0: 'type'})
        df_types = df_types[df_types.type.str.contains("mixed")]
        #This df types logic will ensure the sweetviz analysis runs as expected
        if len(df_types) > 0:
            columns_to_drop = df_types.column.tolist()
            str = ""
            for i in range(len(columns_to_drop)):
                str = str + columns_to_drop[i] + " "
            print("Sweetviz was run, however the following columns were dropped"+ str)
            df = df.drop(columns = columns_to_drop, errors= "ignore")

        cwd = os.getcwd()
        path = cwd + "/data/reports/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        if self.y_var != "None":
            my_report = sv.analyze(df, target_feat= self.y_var)
            my_report.show_html(path + self.sweetviz_name)

        else:
            my_report = sv.analyze(df)
            my_report.show_html(path + self.sweetviz_name)


    def get_pandas_profiling(self) -> None:
        """
        Creates an html file using the pandas profiling library
        """
        profile = pp.ProfileReport(self.df, title = "Exploratory Data Analysis")
        cwd = os.getcwd()
        path = cwd + "/data/reports/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        profile.to_file(output_file = path + "pandas_profiling.html")


    def get_eda(self):
        """
        Prints out some useful eda information to terminal
        """ 
        print("---------------------------------------------------------------------------------")
        print("Total Columns:" + str(len(self.df.columns)))
        print("Number of Categorical Features: ", len([x for x in self.df.select_dtypes(include=['object'])]))
        print("Number of Numeric Features:", len([x for x in self.df.select_dtypes(exclude=['object'])]))
        print("Column typing:")
        print(self.df.dtypes)
        print("---------------------------------------------------------------------------------")
        print("Head of the Data:")
        print(self.df.head())
        print("Describe Summary:")
        print(self.df.describe())

    def print_value_counts(self):
        """
        Prints out the value counts for each column in the dataframe
        """
        for i in self.df.columns:
            print(pd.DataFrame(self.df[i].value_counts(dropna = False)))

    def get_pandas_dtypes(self):
        """ 
        This function returns the pandas dtypes of each column in df
        This is particularly useful for finding mixted type columns and
        can help spot when extra pre-processing will need to be done

        Returns:
            df: uses self.df argument and applies infer_dtype to this
        """
        df = pd.DataFrame(self.df.apply(pd.api.types.infer_dtype, axis=0)).reset_index().rename(columns={'index': 'column', 0: 'type'}) 
        print(df)
        return df

    def run_eda(self):
        """
        A runnable eda function that will run all the eda functions based on an input config
        """
        config = self.config
        if self.config == None:
            print("Config not set! Please set config and try again")
            return

        if "get_missing_column_values" in config:
            if config["get_missing_column_values"] == "True":
                self.get_missing_column_values()
            # This might return out of the whole function
            # Needs checking
        if "get_pandas_dtypes" in config:
            if config["get_pandas_dtypes"] == "True":
                self.get_pandas_dtypes()
        if "get_sweetviz" in config:
            if config["get_sweetviz"] == "True":
                self.get_sweetviz()
        if "get_pandas_profiling" in config:
            if config["get_pandas_profiling"] == "True":
                self.get_pandas_profiling()
        if "get_eda" in config:
            if config["get_eda"] == "True":
                self.get_eda()
        return


   # TODO: Implement some column checking code
   # So idea is to use like check_int on say a year column which would check but if it was "correct"
   # This would end up being a sub-class to contain just column checking info and then get imported into this HEDA class


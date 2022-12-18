import json
import os
import pandas as pd
import pickle

class HDataLoader:

    def __init__(self):
        pass

    def load_a_json(self, location: os.path) -> dict:
        """reads a .json file from disk"""
        if self.check_for_a_file(location):
            with open(location) as file:
                data = json.load(file)
            return data
        else:
            print(location)
            raise ValueError(f"file at {location} not found, try another path")

    def load_a_csv(self, location: os.path) -> dict:
        
        if self.check_for_a_file(location):
            with open(location) as file:
                data = pd.read_csv(file)
            return data
        else:
            raise ValueError(f"file at {location} not found, try another path")

    def pickle_model(self, params, file: str) -> None:
        #Save the object as a pickled file
        f = open(file, "wb")
        pickle.dump(params,f)
        f.close()

    def import_pickled_model(self, location: os.path):    

        if self.check_for_a_file(location):
            with open(location) as file:
                infile = open(file,'rb')
                pickled_model = pickle.load(infile)
                infile.close()
                return pickled_model
        else:
            raise ValueError(f"file at {location} not found, try another path")

    def load_a_xlsx(self, location: os.path) -> dict:
        
        if self.check_for_a_file(location):
            with open(location) as file:
                data = pd.read_excel(file)
            return data
        else:
            raise ValueError(f"file at {location} not found, try another path")

    @staticmethod
    def write_a_json(dictionary: dict, file_path: os.path):
        """writes out a dictionary to a .json object
        make sure your location is pathlike"""

        with open(file_path, "w") as file:
            json.dump(dictionary, file, indent=1)

    @staticmethod
    def check_for_a_file(file_path_str: str) -> bool:
        """returns true or false if a file exists at the specified path"""
        return os.path.exists(os.path.abspath(file_path_str))

    def load_a_text_file(self, location: os.path, sep: str = ",") -> pd.DataFrame:
        """reads a csv file"""

        if self.check_for_a_file(location):
            df = pd.read_csv(location, sep=sep)
            return df
        else:
            raise ValueError(f"file at {location} not found, try another path")

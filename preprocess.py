import utils
import pandas as pd
# This drops the columns that contain a lot of NAs and does some basic feature engineering
modelling_data = pd.read_csv("data/modelling_data.csv")
modelling_data[["Smoker", "Joint?"]] = modelling_data[["Smoker", "Joint?"]].applymap(utils.no_yes_aligner)
#Changing to indicator as makes more sense either youve been in debt or you havent
modelling_data["EF (No. of people not same surname at current address)"] = modelling_data["EF (No. of people not same surname at current address)"].apply(utils.ND_updator)
modelling_data["ND (Months since last CCJ)"] = modelling_data["ND (Months since last CCJ)"].apply(utils.ND_updator)
modelling_data["Sale"] = modelling_data["Sale"].apply(utils.y_aligner)
#One person is missing data for each of Smoker and Person1 age, since it's just one person I'll drop it
modelling_data = modelling_data.dropna()
modelling_data.to_csv("data/modelling_data_clean.csv", index = False)

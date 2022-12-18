import pandas as pd
import utils

"""
Assumptions/General Comments:

{ND} refers to not disclosed
For Numeric columns this is the -999997.0 value
For str (object) columns this is {NA}
Having the columns like this causes issues and steps to rectify:
There are 715 names which null across the bought post Score questions
There are then 825 in some instances and we can assume that the -999997.00 values are the alternative that map to this
9999 score in JF (SCORE) is also associated with these 
X and XX probably the same

I would have thought a date field would be a good indicator of a sale as there is likely to be data drift in insurance data especially around 2008:
Reason being life insurance is often taken out to cover the cost of a morgage or house costs if one of the partners dies
2008 was the start of the financial crisis and the housing market crashed,
as such people may have been worried about their jobs/ability to pay morgage rates and taken out life insurance
"""

#EDA Steps:
data = pd.read_excel("data/MockDataSet1.xlsx").drop(columns= "Unnamed: 0", errors= "ignore")
#Checking Quotes
utils.print_value_counts(data)
#Dropping duplicate rows from a few quotes
data = data.drop_duplicates()

##EDA section
utils.get_eda(data)
utils.get_missing_column_values(data)
print(utils.get_missing_column_values(data))
#Looking at the 110 extras - contains Nob/URB and LSB rest are null or -999997.0
data[(data["WGB (No. of other addresses held)"].isnull() ) & (data["NOB (Property group)"].notnull())]
data["X (Months same person on ER at current address)"].value_counts()


# List of the NA/columns, note will keep EF (No. of people not same surname at current address) in there as it doesn't break sweetviz
# and the contains the 825 values which I'll keep as a binary indicator colum
cols = ['WGB (No. of other addresses held)',
       'X (Months same person on ER at current address)',
       'EF (No. of people not same surname at current address)',
       'NOB (Property group)', 'URB (Income group)',
       'LSB (Regional banded house price band)', 'BB (Number of CCJs)',
       'ND (Months since last CCJ)', "QuoteRef"]

modelling_data = data.drop(columns = cols, errors= "ignore")
utils.get_sweetviz(modelling_data, "Sale") # The columns with invalid {} results break sweetviz interestingly
# Premium and commission are highly correlated so dropping one of them
modelling_data.drop(columns="GrossCommission").to_csv("data/modelling_data.csv", index = False)









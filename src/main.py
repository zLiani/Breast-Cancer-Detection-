import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df) # provides count, mean, std, min, max, etc of data



def clean_data(df, threshold):
    # If more than 5% of the dataset contains rows with missing data, (or a row is more than half null)
    # replace that data with the mean of the column,
    # if less than 5% is corrupt then delete those rows with .dropna()
    missing_in_column = df.isnull().sum() / len(df) * 100
    print ("missing data count:", missing_in_column)
    return df
#new commit

clean_data(df, 0.05)
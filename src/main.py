import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df) # provides count, mean, std, min, max, etc of data

def clean_data(df, threshold):
    # If more than 5% of the dataset contains rows with missing data, (or a row is more than half null)
    # replace that data with the mean of the column,
    # if less than 5% is corrupt then delete those rows with .dropna()
    missing_percentage = df.isnull().mean() * 100

    missing_data_columns = missing_percentage[missing_percentage > threshold].index

    for missing_index in missing_data_columns:
        # if a column is more than half empty, drop column
        if df[missing_index].isnull().mean() > 0.5:
            df.drop(missing_index, axis=1, inplace=True)
        # if column has < 50% but > 5% missing data, fill in mean
        elif df[missing_index].isnull().mean() > 0.05:
            mean_values = df[missing_index].mean()
            df[missing_index].fillna(mean_values, inplace=True)

    return df

clean_data(df, 0.05)
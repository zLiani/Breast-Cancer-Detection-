import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

#print(df) # provides count, mean, std, min, max, etc of data

def clean_data(df, threshold):
    # If more than 5% of the dataset contains rows with missing data, (or a row is more than half null)
    # replace that data with the mean of the column,
    missing_percentage = df.isnull().mean() * 100

    missing_data_columns = missing_percentage[missing_percentage > threshold].index

    for missing_index in missing_data_columns:
        # if a column is more than half empty, drop column
        if df[missing_index].isnull().mean() > 0.5:
            df.drop(missing_index, axis=1, inplace=True)
        # if column has < 50% missing data, fill in mean
        else:
            mean_values = df[missing_index].mean()
            df[missing_index].fillna(mean_values, inplace=True)

    return df

clean_data(df, 0.05)

# xSet is feature vector, ySet is the target
xSet = df.drop('target', axis=1)
ySet = df['target']

# Split data into Training, Testing, and Validation (60%-20%-20%)
x_train, x_test, y_train, y_test = train_test_split(xSet, ySet, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
forestFit = rf.fit(x_train,y_train)
y_valPrediction = rf.predict(x_val)
val_accuracy = accuracy_score(y_val, y_valPrediction)
feature_report = classification_report(y_val, y_valPrediction)
# Use test data after training and validating the model
print(feature_report)
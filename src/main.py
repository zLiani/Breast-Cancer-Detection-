import pandas as pd
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
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
rf.fit(x_train,y_train)
y_valPrediction = rf.predict(x_val)
val_accuracy = accuracy_score(y_val, y_valPrediction)
feature_report_val = classification_report(y_val, y_valPrediction)

# Use test data after training and validating the model
y_testPrediction = rf.predict(x_test)
test_accuracy = accuracy_score(y_test, y_testPrediction)
feature_report_test = classification_report(y_test, y_testPrediction)

# Print Classification Reports and accuracy for validation and test datasets
print(f'Validation Accuracy: {val_accuracy:.2f}')
print("Validation Data Classification Report:")
print(feature_report_val)
print(f'Test Accuracy: {test_accuracy:.2f}')
print("Test Data Classification Report:")
print(feature_report_test)

# Visualize the dataset and the results (maybe add gini impurity, other criteria)
fn=xSet
cn=ySet
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
plot_tree(rf.estimators_[0], feature_names = fn.columns,class_names=["Malignant", "Benign"],filled = True)
fig.savefig('RFC_tree1.png')
#print(feature_report)
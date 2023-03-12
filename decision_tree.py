import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

# Convert a categorical variable to numerical values
encoder = LabelEncoder()
data_train['workclass'] = encoder.fit_transform(data_train['workclass'])
data_train['marital-status'] = encoder.fit_transform(data_train['marital-status'])
data_train['occupation'] = encoder.fit_transform(data_train['occupation'])
data_train['relationship'] = encoder.fit_transform(data_train['relationship'])
data_train['race'] = encoder.fit_transform(data_train['race'])
data_train['gender'] = encoder.fit_transform(data_train['gender'])
data_train['native-country'] = encoder.fit_transform(data_train['native-country'])
data_test['workclass'] = encoder.fit_transform(data_test['workclass'])
data_test['marital-status'] = encoder.fit_transform(data_test['marital-status'])
data_test['occupation'] = encoder.fit_transform(data_test['occupation'])
data_test['relationship'] = encoder.fit_transform(data_test['relationship'])
data_test['race'] = encoder.fit_transform(data_test['race'])
data_test['gender'] = encoder.fit_transform(data_test['gender'])
data_test['native-country'] = encoder.fit_transform(data_test['native-country'])

# Set x_train and X_test to contain columns col1 and col2 of the DataFrame
x_train = data_train[
    ['age', 'workclass', 'fnlwgt', 'educational_num', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
x_test = data_test[
    ['age', 'workclass', 'fnlwgt', 'educational_num', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]

# Set y_train and y_test to contain the target variable of the DataFrame
y_train = data_train['income']
y_test = data_test['income']

# Initialize a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(x_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

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
data_test['workclass'] = encoder.fit_transform(data_test['workclass'])

# Set X_train and X_test to contain columns col1 and col2 of the DataFrame
X_train = data_train[['age', 'workclass']]
X_test = data_test[['age', 'workclass']]
print(X_train)
print(X_test)

# Set y_train and y_test to contain the target variable of the DataFrame
y_train = data_train['income']
y_test = data_test['income']

# Initialize a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

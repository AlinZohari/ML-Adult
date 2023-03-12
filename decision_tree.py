import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
data = pd.read_csv('data_train.csv')

# Convert a categorical variable to numerical values
encoder = LabelEncoder()
data['workclass'] = encoder.fit_transform(data['workclass'])

# Set X_train to contain columns col1 and col2 of the DataFrame
X_train = data[['age', 'workclass']]

# Set y_train to contain the target variable of the DataFrame
y_train = data['income']

# Initialize a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
#y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy}")

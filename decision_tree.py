import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from graphviz import Source

# Load the data from the CSV file
data_train_val = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

# Convert a categorical variable to numerical values
encoder = LabelEncoder()
data_train_val['workclass'] = encoder.fit_transform(data_train_val['workclass'])
data_train_val['marital-status'] = encoder.fit_transform(data_train_val['marital-status'])
data_train_val['occupation'] = encoder.fit_transform(data_train_val['occupation'])
data_train_val['relationship'] = encoder.fit_transform(data_train_val['relationship'])
data_train_val['race'] = encoder.fit_transform(data_train_val['race'])
data_train_val['gender'] = encoder.fit_transform(data_train_val['gender'])
data_train_val['native-country'] = encoder.fit_transform(data_train_val['native-country'])
data_test['workclass'] = encoder.fit_transform(data_test['workclass'])
data_test['marital-status'] = encoder.fit_transform(data_test['marital-status'])
data_test['occupation'] = encoder.fit_transform(data_test['occupation'])
data_test['relationship'] = encoder.fit_transform(data_test['relationship'])
data_test['race'] = encoder.fit_transform(data_test['race'])
data_test['gender'] = encoder.fit_transform(data_test['gender'])
data_test['native-country'] = encoder.fit_transform(data_test['native-country'])

# Set x_train and x_test to contain columns of the DataFrame
feature_names = ['age', 'workclass', 'fnlwgt', 'educational_num', 'marital-status', 'occupation', 'relationship',
                 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Split training data into real training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data_train_val[feature_names], data_train_val['income'], test_size=0.2, random_state=42)

# Define range of values for maximum depth of tree
max_depth_values = range(1, len(feature_names))

# Loop over values of max_depth and calculate cross-validation score for each
cv_scores = []
for max_depth in max_depth_values:
    clf = DecisionTreeClassifier(max_depth=max_depth)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

# Choose value of max_depth that gives best cross-validation score
best_max_depth = max_depth_values[np.argmax(cv_scores)]

print(best_max_depth)

# Train final model using best value of max_depth
final_clf = DecisionTreeClassifier(max_depth=best_max_depth)
final_clf.fit(X_train, y_train)

# Evaluate performance on validation set
val_score = final_clf.score(X_val, y_val)



# Test the tree using the test set
x_train = data_train_val[feature_names]
x_test = data_test[feature_names]

# Set y_train and y_test to contain the target variable of the DataFrame
y_train = data_train_val['income']
y_test = data_test['income']

# Initialize a decision tree classifier
clf = DecisionTreeClassifier(max_depth=best_max_depth)

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(x_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Visualize the decision tree
export_graphviz(clf, out_file="tree.dot", feature_names=feature_names, filled=True,
                rounded=True, special_characters=True)

with open("tree.dot") as f:
    dot_graph = f.read()

graph = Source(dot_graph)
graph.format = "png"
graph.render("decision_tree", view=True)
# dot: graph is too large for cairo-renderer bitmaps. Scaling by 0.194084 to fit






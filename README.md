# Supervised Machine Learning

A collaborative binary classification of supervised machine learning algorithm to an [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) from UCI Machine Learning Repository predicting whether an adult earns over $50,000 per year.

## Abstract
The Adult dataset is a popular dataset available on the UCI Machine Learning Repository that contains
demographic information about individuals, including features such as age, education level, marital status,
occupation, race, sex, and native country. The dataset was from the 1994 United States Census Bureau data and
contains more than 32,000 observations.
The classification task associated with this dataset is to predict whether an adult earns over $50,000 per year,
which is represented by the target variable "income." This task is usually referred to as a binary classification
problem since there are only two possible outcomes: earning less than $50,000 per year or earning more than
$50,000 per year.
This dataset is often used as a benchmark for binary classification tasks because it is a well-known and publicly
available dataset that is representative of real-world problems. Additionally, the dataset is relatively large and
contains a diverse range of features that may be relevant to predicting an individual's income level.
The relevance of applying supervised machine learning algorithms to the Adult data set lies in its ability to provide
insights into factors that correlate with higher income levels, such as education, occupation, age, race, and gender.
This task also has practical applications in fields like credit risk assessment, targeted marketing, and fraud
detection. Additionally, it can contribute to the advancement of research in the field of machine learning and data
science.
We are trying to solve a binary classification task, where the target variable is the income class (<=50K or >50K)

This project includes family algorithm, and its learning algorithm as follows:

- [Decision Trees](https://github.com/AlinZohari/ML-Adult/blob/main/decision_tree.ipynb): classification tree using CART algorithm. 
- [Instance-based Learning](https://github.com/AlinZohari/ML-Adult/blob/main/instance_based-kNN.ipynb): k-Nearest Neighbors 
- [Neural Networks](https://github.com/AlinZohari/ML-Adult/blob/main/NeuralNetwork.ipynb): Feedforward Neural Network, FNN 
- [Bayesian Learning](https://github.com/AlinZohari/ML-Adult/blob/main/ByesLearning.ipynb): GuassianNB, CategoricalNB, BernoulliNB 
- [Model Ensembles](https://github.com/AlinZohari/ML-Adult/blob/main/model_ensemble.ipynb): Stacking (voting mechanism)

Model Performances Measuring

- Performance is measured by accuracy, precision, recall, F1-score, and AUC-ROC.
- Performance is also measured by the running time to compute the algorithm.
- Min train time.
- Visualizations: Use visualizations such as confusion matrices, ROC curves, and precision-recall
curves to gain insights into the model's performance. These visualizations can help identify the
strengths and weaknesses of the model and provide guidance for improving it.

## File structure in this repository
The repository includes the Jupyter notebook files of all tasks, a data folder containing the adult data and the
output folder containing any outputs from our code (e.g. models and txt files).
### The main branch consists of 8 Jupyter notebook files:
  1. [DataCleaning.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/DataCleaning.ipynb)
  2. [Data_Exploration_Visualisation.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/DataCleaning.ipynb)
  3. [decision_tree.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/decision_tree.ipynb)
  4. [instance_based-kNN.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/instance_based-kNN.ipynb)
  5. [ByesLearning.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/ByesLearning.ipynb)
  6. [NeuralNewwork.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/NeuralNetwork.ipynb)
  7. [model_ensemble.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/model_ensemble.ipynb)
  8. [model_comparison.ipynb](https://github.com/AlinZohari/ML-Adult/blob/main/model_comparison.ipynb)
### Other files in the main branch includes files needed by decision_tree.ipynb:
  1. [decision_tree](https://github.com/AlinZohari/ML-Adult/blob/main/decision_tree)
  2. [decision_tree.png](https://github.com/AlinZohari/ML-Adult/blob/main/decision_tree.png)
  3. [tree.dot](https://github.com/AlinZohari/ML-Adult/blob/main/tree.dot)
### The main branch also includes 2 folder which are:
  1. [data](https://github.com/AlinZohari/ML-Adult/tree/main/data)
  2. [output](https://github.com/AlinZohari/ML-Adult/tree/main/output)

## Results
We can compare the following indicators of each algorithm. For accuracy, precision, recall and F1-score, we use
the macro average value from the classification report. The values of Running time and AUC-ROC were
calculated in each task.

| ALGORITHM              | ACCURACY | PRECISION | RECALL | F1-SCORE | RUNNING TIME (s) | AUC-ROC |
|------------------------|----------|-----------|--------|----------|------------------|---------|
| DECISION TREE          | 0.85     | 0.80      | 0.78   | 0.79     | 0.1007           | 0.7465  |
| INSTANCE-BASED LEARNING| 0.84     | 0.79      | 0.75   | 0.76     | 10.6831          | 0.7463  |
| BAYESIAN LEARNING      | 0.83     | 0.76      | 0.76   | 0.76     | 0.0225           | 0.7647  |
| NEURAL NETWORK         | 0.85     | 0.81      | 0.76   | 0.78     | 0.6456           | 0.7554  |
| MODEL ENSEMBLE         | 0.85     | 0.80      | 0.78   | 0.79     | 10.8409          | 0.7768  |

From the table we can easily cmpare the performance of each model:
Accuracy: Measures the proportion of correct predictions made by the classifier. In this table, decision tree,
neural networks, ensemble learning, and decision trees have the highest accuracy (0.85) and Bayesian learning
has the lowest accuracy (0.83).

*Precision*: Measures the proportion of samples for which the classifier correctly predicted a positive class. Neural
networks performed the best in terms of precision (0.81) and Bayesian learning performed the worst (0.76).

*Recall*: A measure of the proportion of samples with an actual positive class that are correctly predicted as a
positive class. Ensemble learning and decision tree was the best performer in terms of recall (0.78), while instancebased
learning was the worst performer (0.75).

*F1 score*: This is the average of precision and recall and is used to assess the performance of the model. Ensemble
model and decision trees had the highest F1 score (0.79), while instance-based learning and Bayesian learning
had the lowest F1 score (0.76).

*Running time*: This indicates the time it takes for the model to make predictions. Bayesian learning had the
shortest run time (0.0225 seconds), while the ensemble learning had the longest run time (10.8409 seconds).

*AUC-ROC*: measures the performance of the binary classifier at different thresholds. Ensemble learning had the
highest AUC-ROC (0.7768), while Instance-Based learning, kNN had the lowest AUC-ROC (0.7463).

In taken together, the relevant evaluation indicators for each model do not differ significantly. Since we chose to
train the models with a primary focus on the AUC-ROC metric, ensemble learning was better than the other
models in all cases on that metric.

## Conclusion

The **decision tree** has an advantage in terms of running time, because the mechanism is rather simple. However,
the AUC-ROC score is not the best. While decision trees are good at handling categorical features and capturing
non-linear relationships, they may not be as effective in modelling complex interactions between features.
Decision trees make splits based on one feature at a time, which can lead to underfitting or overfitting in some
cases. Since there are probably complex interactions between multiple features in the Adult Dataset, decision
trees may not be able to capture them effectively.

**Instance-based learning** takes the longest to run, as it is computationally expensive due to the requirement of
calculating the distance between test data and training data during the prediction phase. To reduce the computation
time, one can try using approximate nearest neighbour search techniques such as locally sensitive hashing (LSH)
or tree structures. This method will enable quicker retrieval of the nearest neighbours, improving the efficiency
of the prediction phase.

**Bayesian learning** Although the scores are slightly lower than all other algorithms, the running time is significant
short. This is due to the simplicity of calculation required from Naives Bayes classifiers. This is the advantage of
the applied statistical theory in Bayes learning but it’s also the limitation. Since the assumption of independence
between features affects the performance of the model. More complex Bayesian models such as Bayesian
networks can be tried to capture the dependencies between features.

The **neural network** performed well in terms of accuracy, precision and F1 scores, but the AUC-ROC was
relatively low. This may be because the models are still slightly overfitted to the training data. The search for the
best combination of hyperparameters to improve the model can continue.

**Model Ensemble** performed best on various metrics, suggesting that performance can be effectively improved
by combining multiple models. It is probably because it considers predictions from all base models, thus take
advantages of all models. However, the result is not very convincing, as it is only better a bit than the second best
(0.03%). Further tuning might be needed to improve the result. Also, more experimentation with different
integration methods, such as Bagging, Boosting or Stacking, as well as trying to combine more different types of
base learners, could be used to further improve performance.

In conclusion, compared with AUC_ROC Score, the performance of the decision tree is very close to the ensemble.
It is the best model except for the ensemble. This is also reflected when tuning parameters of the ensemble model,
that it gives more accurate results when assigning higher weights to the predictions of decision trees. Other
indicators are the same as the ensemble model, except for running time. The running time of the decision tree is
far less than the ensemble model. Therefore, in the case where other indicators are very close, it will be better to
use a decision tree, because it produces a good result in a short period of time.



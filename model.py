import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.auto_encoder import AutoEncoder

contamination = 0.05  # percentage of outliers
n_train = 500  # number of training points
n_test = 500  # number of testing points
n_features = 25  # number of features
x_train, x_test, y_train, y_test = generate_data(
    n_train=n_train,
    n_test=n_test,
    n_features=n_features,
    contamination=contamination,
    random_state=123
)

x_train_pd = pd.DataFrame(x_train)
x_train_pd.head()

plt.scatter(x_train_pd[0], x_train_pd[1], c=y_train, alpha=0.8)
plt.title('Scatter Plot')
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()

#  Autoencoder with two hidden layers, each hidden layer with two neurons.
autoencoder = AutoEncoder(contamination=0.05, hidden_neurons=[2, 2])
autoencoder.fit(x_train)

#  Training data
y_train_scores = autoencoder.decision_function(x_test)
y_train_pred = autoencoder.predict(x_train)

#  Testing data
y_test_scores = autoencoder.decision_function(x_test)
y_test_pred = autoencoder.predict(x_test)  # Outlier labels (0 or 1)

#  Determine the auto-generated contamination rate
print('The threshold for the defined contamination rate:', autoencoder.threshold_)


def count_stat(vector):
    # Because it is '0' and '1', we can run a count statistic
    unique, counts = np.unique(vector, return_counts=True)
    return dict(zip(unique, counts))


print('The training data:', count_stat(y_train_pred))
print('The testing data:', count_stat(y_test_pred))

# Because we traditionally don't know the percentage of outliers, we can use the histogram of the outlier score to
# select a reasonable threshold. Take the natural cut in the histogram to determine the score.

plt.figure(figsize=(6, 4), dpi=80)
plt.hist(y_train_scores, bins='auto')  # arguments are passed to np.histogram
plt.title('Outlier score')
plt.show()

threshold = autoencoder.threshold_  # Or other value from the above threshold. The default is 10


def descriptive_stat_threshold(df, pred_score, threshold):
    # Determine how many 0's and 1's there are
    df = pd.DataFrame(df)
    df['Anomaly_Score'] = pred_score
    df['Group'] = np.where(df['Anomaly_Score'] < threshold, 'Normal', 'Outlier')

    # Show the summary statistics
    cnt = df.groupby('Group')['Anomaly_Score'].count().reset_index().rename(columns={'Anomaly_Score': 'Count'})
    cnt['Count %'] = (cnt['Count'].sum()) * 100  # The count and the count %
    stat = df.groupby('Group').mean().round(2).reset_index()  # The average
    stat = cnt.merge(stat, left_on='Group', right_on='Group')  # Put the count and the average together

    return stat


table = descriptive_stat_threshold(x_train, y_train_scores, threshold)
print(table)


#  Generate a confusion matrix
def confusion_matrix(actual, score, threshold):
    actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
    actual_pred['Pred'] = np.where(actual_pred['Pred'] <= threshold, 0, 1)
    cm = pd.crosstab(actual_pred['Actual'], actual_pred['Pred'])

    return cm


confusion_matrix = confusion_matrix(y_train, y_train_scores, threshold)
print(confusion_matrix)

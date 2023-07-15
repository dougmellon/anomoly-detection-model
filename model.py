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
y_training_scores = autoencoder.decision_function(x_test)
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
plt.hist(y_training_scores, bins='auto')  # arguments are passed to np.histogram
plt.title('Outlier score')
plt.show()


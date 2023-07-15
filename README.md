# Anomaly Detection with Autoencoders
The modeling procedure for the model development, assessment, and interpretation of the results:
- Model development
- Threshold determination
- Descriptive statistics of the normal and abnormal groups

With the outlier scores, you will choose a threshold to separate the abnormal observations with high outlier scores 
from normal observations. If any prior knowledge suggests the percentage of anomalies should be no more than 1%, we can
choose a threshold that results in approximately 1% of anomalies.

## The Profile Table of the Normal and Abnormal Groups
The table represents the characteristics of the normal and abnormal groups. It shows the count and count percentage 
of the normal and outlier groups. Make sure to label the features with their feature names for an effective presentation.

The table tells us the following:
- The size of the outlier group: Once a threshold is determined, the size is determined. If the threshold is derived manually from the chart generated and there is no prior knowledge, the size of the statistic becomes a good reference to start with.
- The feature statistics in each group: All the means must be consistent with the domain knowledge. In our case, the means in the outlier group are smaller than those in the normal group.
- The average anomaly score: The average score of the outlier group should be higher than that of the normal group.
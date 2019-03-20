import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
anomalies = []
ind = [] # Anomalies position
# multiply and add by random numbers to get some real values
random_data = np.random.randn(5000)  * 20 + 20

# Function to Detection Outlier on one-dimentional datasets.
def find_anomalies(random_data):
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3

    lower_limit  = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
#    print(lower_limit)
    # Generate outliers
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    random_data = list(random_data)
    for i in anomalies:
        ind.append(random_data.index(i))
    return anomalies, ind

a = (find_anomalies(random_data))
print(a[0])
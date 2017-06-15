# Summer Research on Machine Learning 

Creating anomaly detection algorithms for Autoencoders with Professor Nathalie Japkowicz at American University

The data sets are from UC Irvine Machine Learning Repository and I am using Wisconsin Breast Cancer, as well as Shuttle Data set from NASA

The first_autoencoder.py is a simple autoencoder on the breast cancer data that differentiates between maliganant and benign cancer

Streaming autoencoder uses the shuttle data set which is a time series and tries to differentiate Class 1 from all the other classes. To handle the time series I have implemented a change point detection algorithm which can detect changes to the structure of the data over time.

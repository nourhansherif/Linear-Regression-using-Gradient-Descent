### Necessary Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


### Data Reading & Preprocessing ###

def DataReadingAndVisualization():
    # Date Reading
    data = pd.read_csv('dataset/Fish.csv')
    # View The Data Description
    print('Data Description:')
    print(data.describe())
    # View The Data Content
    print('Data Content:')
    print(data.head())

    # Check if there are any missing values
    print('Check if there are any missing values:')
    print(data.isna().sum())

    return data


# Visualize the presence of Outliers In Data
def OutliersVisualization(data):
    for col in data.columns:
        if col != 'Species':
            data.boxplot(col)
            plt.show()

# Checking The Presence Of Outliers
def OutliersDetectionUsingZScore(data):
    outliers = []
    for col in data.columns:
        if col != 'Species':
            threshold = 3
            mean = np.mean(data[col])
            std = np.std(data[col])
            for feature in data[col]:
                z_score = (feature - mean) / std
                if np.abs(z_score) > threshold:
                    outliers.append(feature)
    return outliers


# Outliers Treatment with median
def OutliersTreatmentWithMedian(data, outliersList):
    for col in data.columns:
        if col != 'Species':
            ## Before
            # OutliersVisualization(data)
            median = np.median(data[col])
            for outlier in outliersList:
                data[col] = np.where(data[col] == outlier, median, data[col])
            ## After
            # OutliersVisualization(data)


# Histogram Distribution Of Data To Check For Scaling
def HistogramDistribution():
    dataNumericalFeatures = data.drop(columns=["Species"])
    fig, axes = plt.subplots(len(dataNumericalFeatures.columns) // 3, 3, figsize=(15, 6))
    counter = 0
    for triaxis in axes:
        for axis in triaxis:
            dataNumericalFeatures.hist(column=dataNumericalFeatures.columns[counter], ax=axis)
            counter += 1


def OutliersDetectionAndTreatment(data):
    ## Before
    # HistogramDistribution()
    outliersList = OutliersDetectionUsingZScore(data)
    print("Outliers from Z-scores method: ", outliersList)
    OutliersTreatmentWithMedian(data, outliersList)
    ## After
    # HistogramDistribution()


# Get Input's Features and True Outputs
def ExtractFeaturesAndTrueOutputs(data):
    X = data.drop(['Weight'], axis=1)
    Y = data['Weight'].values.reshape(-1, 1)
    return X, Y


# Label Encoding
def LabelEncoding(X):
    label_encoder = LabelEncoder()
    X['Species'] = label_encoder.fit_transform(X['Species'])
    return X.values


# Apply Feature Scaling Standardization
def Standardization(X):
    standard_scaler = StandardScaler()
    standardizedX = standard_scaler.fit_transform(X)
    return standardizedX


def DataPreProcessing(data):
    X, Y = ExtractFeaturesAndTrueOutputs(data)
    encodedX = LabelEncoding(X)
    standardizedX = Standardization(encodedX)
    return standardizedX, Y


### Linear Regression using Gradient Descent ###

def Cost(Y, X, theta):
    y_pred = np.dot(X, theta)
    m = Y.shape[0]
    current_cost = sum((y_pred - Y) ** 2) / (2 * m)
    return current_cost


def Gradient(Y, X, theta):
    y_pred = np.dot(X, theta)
    error = y_pred - Y
    m = Y.shape[0]
    return np.dot(np.transpose(X), error) / m


def GradentDescentAlgorithm(X, Y):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    theta = np.zeros([len(X[0]), 1])
    tolerance = 0.000001
    current_cost = 10000
    learning_rate = 0.01
    max_iterations = 1000
    costList = []
    iterationsList = []

    for i in range(1, max_iterations + 1):
        if current_cost < tolerance:
            break
        else:
            gradient = Gradient(Y, X, theta)
            theta = theta - learning_rate * gradient

        current_cost = Cost(Y, X, theta)

        if i % 10 == 0:
            costList.append(current_cost[0])
            iterationsList.append(i)
    print(costList[-1])
    return iterationsList, costList


def GraphPlotting(iterationsList, costList):
    plt.title('Cost Function')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(iterationsList, costList)
    plt.show()


# Data Reading & Preprocessing
data = DataReadingAndVisualization()
OutliersDetectionAndTreatment(data)
X, Y = DataPreProcessing(data)

# Linear Regression using Gradient Descent
iterationsList, costList = GradentDescentAlgorithm(X, Y)
GraphPlotting(iterationsList, costList)
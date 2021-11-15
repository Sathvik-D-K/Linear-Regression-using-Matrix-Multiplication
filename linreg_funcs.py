import numpy as np


path = "E:/spyder Ide Python/Wine quality linreg/winequality-white.csv"
num_train = 3674
num_test = 1224

def load_data(path, num_train):
    #Load the data matrices
 
    data = np.loadtxt(path,delimiter = ";", skiprows=True) 
    
    data_train = data[: num_train,]
    data_test = data[num_train:]
    
    X_train, Y_train = data_train[:, :-1], data_train[:, -1]
    X_test, Y_test = data_test[:, :-1], data_test[:, -1]
    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    """ Fit linear regression model
    Input:
    X: numpy array of shape N x n containing data
    Y: numpy array of shape N containing targets
    Output:
    theta: nump array of shape n + 1 containing weights
           obtained by fitting data X to targets Y
           using linear regression
    """
    
    X = np.hstack((X, np.ones((X.shape[0],1))))
    
    theta = np.linalg.inv(X.T @ X)@(X.T @ Y)
    return theta


def predict(X, theta):
    """
    Input:
    X: numpy array of shape N x n containing data
    theta: numpy array of shape n + 1 containing weights
    Output:
    Y_pred: numpy array of shape N containig predictions
    """
    # TODO
    X = np.hstack((X, np.ones((X.shape[0],1))))
    Y_pred = X @ theta
    return Y_pred


def energy(Y_pred, Y_gt):
    """ Calculate squared error
    Input:
    Y_pred: numpy array of shape N containing prediction
    Y_gt: numpy array of shape N containing targets
    Output:
    se: squared error between Y_pred and Y_gt
    """
    se = np.sum((Y_pred - Y_gt)**2)
    
    return se

"""
# load data matrices
X_train, Y_train, X_test, Y_test = load_data(path, num_train)

theta = fit(X_train, Y_train)
print("Fitted weights:")
print(theta.shape)


# perform prediction on the test set
Y_pred = predict(X_test, theta)
en = energy(Y_pred, Y_test)
print(f"MSE loss on test data: {en / num_test}") # printing the MSE loss

"""
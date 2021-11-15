# -*- coding: utf-8 -*-

from linreg import *

# load data matrices
X_train, Y_train, X_test, Y_test = load_data(path, num_train)

theta = fit(X_train, Y_train)
print("shape of fitted weights ", theta.shape)


# perform prediction on the test set
Y_pred = predict(X_test, theta)
en = energy(Y_pred, Y_test)

print(f"MSE loss on test data: {en / num_test}") # printing the MSE loss
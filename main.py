#  Bethanie Williams
#  CSC 5220 Data Mining and Machine Learning
#  Lab 2 Assignment
#  Date: 8/25/20
#  Professor: Dr. Ismail

########################################################################################
# # Loaded libraries
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
# import random
from sklearn.datasets import make_regression
import pylab
# from scipy import stats
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

#######################################################################################
# # Summarizing Data Sets


def sets_of_data(dataset):

    print("The shape is", dataset.shape)  # shape and checking to see if all data is there
    print()  # dataset.shape shows us how many rows and columns are in the data

    print("Training Set: ")  # defining training set
    x_training = dataset.iloc[:12, :1]  # gets rows until the 12th row and first two columns
    y_training = dataset.iloc[:12, 1:2]
    training_data = (x_training, y_training)
    print(training_data)
    print(x_training)
    print()
    print(y_training)
    print()

    print("Test Set:")  # defining test set
    x_testing = dataset.iloc[:, 2:3]  # gets all rows and next two columns
    y_testing = dataset.iloc[:, 3:4]
    print(x_testing)
    print()
    print(y_testing)
    print()

    print("Validation Set:")  # defining validation set
    x_validation = dataset.iloc[:, 4:5]  # gets all rows and last two columns
    y_validation = dataset.iloc[:, 5:6]
    print(x_validation)
    print()
    print(y_validation)
    print()

    pyplot.scatter(x_training, y_training)
    pyplot.xlabel('x-training')  # labels x-axis
    pyplot.ylabel('y-training')  # labels y-axis
    pyplot.show()  # displays scatter plot

###########################################################################################


def gradient_descent_2(alpha, x, y, num_iterations):
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iterate in range(0, num_iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        j = np.sum(loss ** 2) / (2 * m)  # cost
        print("iter %s | J: %.3f" % (iterate, j))
        gradient = np.dot(x_transpose, loss) / m
        theta = theta - alpha * gradient  # update
    return theta
# # # Evaluating Algorithms and Building Models
#
#
# array = dataset.values  # Split-out validation dataset
# X = array[:, 0:4]
# y = array[:, 4]  # leaves some of the data to be usable for the trained data
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
#
# models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),  # building models
#           ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
#           ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
#           ('SVM', SVC(gamma='auto'))]  # Spot Check Algorithms
#
#
# results = []
# names = []
# for name, model in models:  # evaluates each model in turn
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#
#
# pyplot.boxplot(results, labels=names) # Compare Algorithms
# pyplot.title('Algorithm Comparison')
# pyplot.show()  # displays how the algorithms compared
#
# ######################################################################################
# # # Making and Evaluating Predictions
#
# model = SVC(gamma='auto')  # Make predictions on validation dataset
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)
#
# print()  # Evaluate predictions
# print("Accuracy:", accuracy_score(Y_validation, predictions))
# print("Errors made include:")
# print(confusion_matrix(Y_validation, predictions))
# print("The classification report:")  # shows the breakdown of each class by precision,
# print(classification_report(Y_validation, predictions))  # recall, f1-score and support showing

# #########################################################################################


def main():
    data = read_csv(r'C:\Users\betha\OneDrive\Documents\CSC5220\Lab2_dataset.csv')  # importing csv
    print(data)
    sets_of_data(data)

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                           random_state=0, noise=35)
    m, n = np.shape(x)
    x = np.c_[np.ones(m), x]  # insert column
    alpha = 0.01  # learning rate
    theta = gradient_descent_2(alpha, x, y, 1000)
    print("Theta = ", theta)

    # plot
    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1] * x
    pylab.plot(x[:, 1], y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()
    print("Done!")


if __name__ == "__main__":
    main()


import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!
from logreg import LogisticRegression


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 1 to class 0
kappa = 0.1


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x = x_train
    y = y_train
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)
    clf = LogisticRegression()
    theta = clf.fit(x_train,y_train)
    y_predict = clf.predict(x_valid)
    def report_accuracy(y_valid, y_predict):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(y_valid.shape[0]):
            if y_valid[i] == 0:
                if y_predict[i] < 0.5:
                    TN += 1
                else:
                    FP += 1
            else:
                if y_predict[i] < 0.5:
                    FN += 1
                else:
                    TP += 1
        A = (TP + TN)/(TP + FN + TN + FP)
        A1 = TP/(TP + FN)
        A0 = TN/(TN + FP)
        Abal = (A0 + A1)/2
        return (A,A1,A0,Abal)
    A,A1,A0,Abal = report_accuracy(y_valid,y_predict)
    util.plot(x_valid, y_valid, theta, "plot_naive.png")
    np.savetxt(output_path_naive, y_predict)

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (d): Upsampling minority class
    clf2 = WeightedLogisticRegression()
    theta = clf2.fit(x_train,y_train)
    y_predict = clf2.predict(x_valid)
    A,A1,A0,Abal = report_accuracy(y_valid,y_predict)
    util.plot(x_valid, y_valid, theta, "plot_upsampling.png")
    np.savetxt(output_path_upsampling, y_predict)

class WeightedLogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        w = []
        for i in range(x.shape[0]):
            if y[i] == 0:
                w.append(1)
            else:
                w.append(1/kappa)
        w = np.array(w).reshape((x.shape[0],1))
        def g(w):
            return 1/(1 + np.exp(-w))
        def h(x, theta):
            return g(theta.dot(x.T).reshape(x.shape[0],1))
        def grad_J(x,y,theta):
            n = x.shape[0]
            y = y.reshape(n,1)
            return (-(1+kappa)/(2*n))*np.sum(w*x*(y - h(x,theta)),axis=0)
        def hessian_J(x,y,theta):
            n, _ = x.shape
            return (1+kappa)/(2*n)*np.matmul((w*x).T,(h(x,theta)*(1-h(x,theta)))*x)

        n, d = x.shape
        if self.theta == None:
            self.theta = np.zeros(d)
        iter_count = 0
        while True:
            if iter_count > self.max_iter:
                break
            prev_theta = self.theta
            self.theta = self.theta - self.step_size*np.linalg.inv(hessian_J(x,y,self.theta)).dot(grad_J(x,y,self.theta))
            if np.linalg.norm((self.theta - prev_theta), 1) < self.eps:
                break
            iter_count = iter_count + 1

        return self.theta

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        def g(w):
            return 1/(1 + np.exp(-w))
        def h(x, theta):
            return g(theta.dot(x.T).reshape(x.shape[0],1))
        return h(x, self.theta)

    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')

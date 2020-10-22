import svm
import math
import util
import logreg
import collections
import numpy as np

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of the fitted model, consisting of the
    learned model parameters.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    model = {}
    n,v_size = matrix.shape
    phi_1 = (1 + labels.reshape((1,n)).dot(np.int64(matrix > 0)))/(v_size + labels.dot(np.sum(matrix, axis=1)))
    inv_labels = np.ones(n) - labels
    phi_0 = (1 + inv_labels.reshape((1,n)).dot(np.int64(matrix > 0)))/(v_size + inv_labels.dot(np.sum(matrix, axis=1)))
    phi = np.sum(labels)/n

    model["phi_1"] = phi_1
    model["phi_0"] = phi_0
    model["phi"] = phi
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model (int array of 0 or 1 values)
    """
    # *** START CODE HERE ***
    phi_1 = model["phi_1"]
    phi_0 = model["phi_0"]
    matrix1 = phi_1*np.int64(matrix > 0)
    matrix0 = phi_0*np.int64(matrix > 0)
    phi = model["phi"]
    p_1 = np.exp((np.sum(np.where(matrix1 > 0, np.log(matrix1.astype(np.float64)),matrix1),axis = 1) + np.log(phi)).astype(np.float64))
    p_0 = np.exp((np.sum(np.where(matrix0 > 0, np.log(matrix0.astype(np.float64)),matrix0),axis = 1) + np.log(1 - phi)).astype(np.float64))
    return 1*(p_1 > p_0)
    # *** END CODE HERE ***



def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    indic_dict = {}
    for word in dictionary:
        indicativeness = np.log(model["phi_1"][0][dictionary[word]]/model["phi_0"][0][dictionary[word]])
        indic_dict[word] = indicativeness
    sorted_dict = [k for k, v in sorted(indic_dict.items(), key=lambda item: item[1],reverse = True)]
    return sorted_dict[:5]
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use validation set accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_radius = 0
    best_accuracy = 0
    for radius in radius_to_consider:
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(predictions == val_labels)
        if accuracy > best_accuracy:
            best_radius = radius
            best_accuracy = accuracy

    return best_radius
    # *** END CODE HERE ***

def compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, learning_rates_to_consider):
    """Compute the best logistic regression learning rate using the provided training and evaluation datasets.

    You should only consider learning rates within the learning_rates_to_consider list.
    You should use validation set accuracy as a metric for comparing the different learning rates.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        learning_rates_to_consider: The learning rates to consider

    Returns:
        The best logistic regression learning rate which maximizes validation set accuracy.
    """
    # *** START CODE HERE ***
    best_lr = 0
    best_accuracy = 0
    for learning_rate in learning_rates_to_consider:
        predictions = logreg.train_and_predict_logreg(train_matrix, train_labels, val_matrix, learning_rate)
        accuracy = np.mean(predictions == val_labels)
        if accuracy > best_accuracy:
            best_lr = learning_rate
            best_accuracy = accuracy
    return best_lr
    # *** END CODE HERE ***


def run_naive_bayes(train_matrix, train_labels, test_matrix, test_labels, dictionary, predictions_path):
    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt(predictions_path, naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

# def run_svm():
#     optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

#     util.write_json('spam_optimal_radius', optimal_radius)

#     print('The optimal SVM radius was {}'.format(optimal_radius))

#     svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

#     svm_accuracy = np.mean(svm_predictions == test_labels)

#     print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


#     train_matrix = util.load_bert_encoding('bert_train_matrix.tsv.bz2')
#     val_matrix = util.load_bert_encoding('bert_val_matrix.tsv.bz2')
#     test_matrix = util.load_bert_encoding('bert_test_matrix.tsv.bz2')

#     best_learning_rate = compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.001, 0.0001, 0.00001, 0.000001])

#     print('The best learning rate for logistic regression is {}'.format(best_learning_rate))

#     logreg_predictions = logreg.train_and_predict_logreg(train_matrix, train_labels, test_matrix, best_learning_rate)

#     logreg_accuracy = np.mean(logreg_predictions == test_labels)

#     print('The Logistic Regression model with BERT encodings had an accuracy of {} on the testing set'.format(logreg_accuracy))

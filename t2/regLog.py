import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(z):
    """
    Computes the sigmoid of z (logistic function).

    The sigmoid function maps real-valued numbers to the (0, 1) interval.
    It is commonly used in logistic regression and neural networks.

    :param z: A scalar or numpy array of real numbers. The input to the sigmoid function.
    :return: The sigmoid value for each element of z. The result is between 0 and 1.
    """
    z = np.array(z, dtype=np.float64)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    """
    Computes the cost function (logistic loss function) for logistic regression.

    This function calculates the binary cross-entropy loss (log loss) for the logistic regression model.
    It is used to evaluate how well the model's predictions match the actual target values.

    :param X: The feature matrix (m x n), where m is the number of samples, and n is the number of features.
    :param y: The target vector (m x 1), containing binary labels (0 or 1).
    :param theta: The model parameters (n x 1), representing the weights of the features.
    :return: The computed cost (log loss) value, representing the error between the predicted and actual labels.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-10
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Performs gradient descent to optimize the model parameters (theta).

    Gradient descent is an optimization algorithm used to minimize the cost function
    by adjusting the parameters in the direction of the negative gradient.

    :param X: The feature matrix (m x n), where m is the number of samples, and n is the number of features.
    :param y: The target vector (m x 1), containing the true labels.
    :param theta: The initial model parameters (n x 1), which are updated during the training process.
    :param learning_rate: The step size that controls how much to adjust the parameters in each iteration.
    :param num_iterations: The number of iterations to run the gradient descent algorithm.
    :return: The optimized model parameters (theta) and the history of the cost function values during training.
    """
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta = theta - learning_rate * gradient.astype(float)
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def predict(X, theta):
    """
    Makes predictions using the learned logistic regression model.

    This function applies the sigmoid function to the linear combination of features
    and model parameters, and classifies the output as 1 if the probability is >= 0.5,
    and 0 otherwise.

    :param X: The feature matrix (m x n), where m is the number of samples, and n is the number of features.
    :param theta: The optimized model parameters (n x 1).
    :return: A list of predicted class labels (1 or 0) for each sample in the feature matrix X.
    """
    probabilities = sigmoid(np.dot(X, theta))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]

def feature_importance(theta, feature_names):
    """
    Ranks the features based on the magnitude of their coefficients in the model.

    The magnitude of the coefficients (theta) indicates the importance of each feature in predicting the target.
    Larger absolute values of theta correspond to features that have a stronger impact on the prediction.

    :param theta: The model parameters (n x 1), where each value represents the weight of a feature.
    :param feature_names: A list of feature names corresponding to the model parameters.
    :return: A sorted list of tuples, where each tuple contains a feature name and its importance (absolute value of its coefficient).
    """
    importance = [(feature, abs(coeff)) for feature, coeff in zip(feature_names, theta)]
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    return importance

dataset_path = "tourism_dataset.csv"
data = pd.read_csv(dataset_path)

country = "India"
data = data[data['Country'] == country]

categories = pd.get_dummies(data['Category'], prefix='Category')
data = pd.concat([data, categories], axis=1)

data['Revenue_Per_Visitor'] = data['Revenue'] / data['Visitors']
data['Target'] = (data['Revenue_Per_Visitor'] > data['Revenue_Per_Visitor'].median()).astype(int)

feature_columns = categories.columns.tolist() + ['Visitors']
X = data[feature_columns].values
y = data['Target'].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

feature_names = ["Intercept"] + feature_columns

theta = np.zeros(X_train.shape[1])

learning_rate = 0.1
num_iterations = 1000

theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

predictions = predict(X_test, theta)

accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

importance = feature_importance(theta, feature_names)

category_importance = [(feature, coeff) for feature, coeff in importance if 'Category' in feature]
category_importance = sorted(category_importance, key=lambda x: x[1], reverse=True)

print("\nCategory Ranking:")
for rank, (category, coeff) in enumerate(category_importance, start=1):
    print(f"{rank}. {category}: {coeff:.4f}")

from sklearn.model_selection import KFold


def cross_validate(X, y, num_folds, learning_rate, num_iterations):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))

        theta = np.zeros(X_train.shape[1])

        theta, _ = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

        predictions = predict(X_val, theta)
        accuracy = np.mean(predictions == y_val) * 100
        accuracies.append(accuracy)

    return accuracies


num_folds = 5
cross_val_accuracies = cross_validate(X, y, num_folds, learning_rate, num_iterations)
mean_accuracy = np.mean(cross_val_accuracies)
cross_val_accuracies = [float(acc) for acc in cross_val_accuracies]

print(f"\nCross-validation Accuracies: {cross_val_accuracies}")
print(f"Mean Accuracy: {mean_accuracy:.2f}%")

# import matplotlib.pyplot as plt
#
# correct_predictions = np.sum(np.array(predictions) == y_test)
# incorrect_predictions = len(y_test) - correct_predictions
#
# labels = ['Correct', 'Incorrect']
# sizes = [correct_predictions, incorrect_predictions]
# colors = ['#4CAF50', '#FF6F61']
#
# plt.figure(figsize=(6, 6))
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.title('Classification Results: Correct vs. Incorrect Predictions')
# plt.show()



# # Plot category importance
# categories = [item[0] for item in category_importance]
# coefficients = [item[1] for item in category_importance]
#
# plt.figure(figsize=(10, 6))
# plt.barh(categories, coefficients, color='skyblue')
# plt.xlabel('Coefficient Magnitude')
# plt.ylabel('Category')
# plt.title('Category Importance Based on Logistic Regression Coefficients')
# plt.gca().invert_yaxis()  # Invert y-axis to show the highest-ranked category at the top
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
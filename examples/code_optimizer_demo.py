from mainera.src.code_parallelizer import parallelize_code

python_code = """
def LogesticRegression(X, y, learning_rate=0.01, num_iterations=1000):
    import numpy as np
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = 1 / (1 + np.exp(-linear_model))
        dw = (1 / m) * np.dot(X.T, (y_predicted - y))
        db = (1 / m) * np.sum(y_predicted - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias
"""

cpp_code = parallelize_code(python_code, "converted_code.cpp")

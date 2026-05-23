if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    y_train = df_train["ViolentCrimesPerPop"]
    X_train = df_train.drop("ViolentCrimesPerPop", axis = 1)

    y_test = df_test["ViolentCrimesPerPop"]
    X_test = df_test.drop("ViolentCrimesPerPop", axis = 1)

    lambda_max = 2 * max(np.abs(np.matmul(X_train.T, y_train - np.mean(y_train))))
    _lambda = np.array([lambda_max])
    weights = np.zeros((1, X_train.shape[1]))
    num_nonzero = np.array([0])

    while (_lambda[_lambda.size - 1] >= 0.01):
        w_hat = train(X_train.values, y_train.values, _lambda[-1], start_bias = 0, start_weight = weights[-1])[0]
        num_nonzero = np.append(num_nonzero, np.count_nonzero(w_hat))
        weights = np.vstack([weights, w_hat])
        _lambda = np.append(_lambda, _lambda[-1] / 2)

    # Plot number of nonzero weights along regularization path (c)
    plt.plot(_lambda, num_nonzero)
    plt.title("Number of Nonzero Weights Along Regularization Path")
    plt.xlabel("Log Lambda")
    plt.ylabel("Nonzero Weight Count")
    plt.xscale('log')
    plt.savefig("regularization_path_plot.png")
    plt.show()

    plt.cla()
    plt.clf()

    # Plot regularization paths of weights (d)
    vars = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    idx = X_train.columns.get_indexer(vars)

    lines = plt.plot(_lambda, weights[:, idx])
    plt.title("Regularization Path of Selected Weights")
    plt.xlabel("Log Lambda")
    plt.ylabel("Weights")
    plt.xscale('log')
    plt.legend(lines, vars)
    plt.savefig("regularization_path_weights.png")
    plt.show()

    plt.cla()
    plt.clf()

    # Plot squared error against lambda (e)
    sse_train = np.zeros(_lambda.size)
    sse_test = np.zeros(_lambda.size)
    for i in range(_lambda.size):
        y_hat_train = np.matmul(X_train, weights[i, :])
        y_hat_test = np.matmul(X_test, weights[i, :])
        sse_train[i] = np.matmul((y_train - y_hat_train).T, y_train - y_hat_train)
        sse_test[i] = np.matmul((y_test - y_hat_test).T, y_test - y_hat_test)

    sse = np.vstack((sse_train, sse_test))
    lines = plt.plot(_lambda, sse.T)
    plt.title("Regularization Path of Squared Error")
    plt.xlabel("Log Lambda")
    plt.ylabel("SSE")
    plt.xscale('log')
    plt.legend(lines, ["Train SSE", "Test SSE"])
    plt.savefig("regularization_path_sse.png")
    plt.show()

    # Extreme weights for fixed lambda (f)
    w_hat = train(X_train.values, y_train.values, _lambda = 30)[0]
    print(f"The maximum w_hat is {np.max(w_hat)}, which is on {X_train.columns[np.argmax(w_hat)]}")
    print(f"The minimum w_hat is {np.min(w_hat)}, which is on {X_train.columns[np.argmin(w_hat)]}")


if __name__ == "__main__":
    main()

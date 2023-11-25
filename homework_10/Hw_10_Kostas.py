import numpy as np
import matplotlib.pyplot as plt

x_i = np.arange(-2, 3, 1)
y_i = np.array([-1, -0.1, 0.2, 0.1, 0.5])


def Phi(x, N, poly=True):
    """
    Function that constructs the Phi vector space.
    In this spcific case, the feature space is the space of polynomials up to order
    Additionally, another feature space is tested (sinusoidal feature space).
    :param x:
    :param N: the dimension of the feature space
    :return:
    """
    if poly:
        return np.array([x ** i for i in range(N)])
    else:
        return np.array([np.sin(x)**i for i in range(N)])


def GaussianProcess(sigma, N, x_obs, y_obs, x_star, poly=True):
    """
    Function that computes the Gaussian process regression. The function calculates the posterior distribution of the
    model weights, and the predictive distribution of f*, given an input vector of new points (x*)

    :param sigma: the standard deviation of the model error y_i = f(x_i) +e_i
    :param N: the dimension of the feature space
    :param x_obs: the x_i training data
    :param y_obs: the y_i training data
    :param x_star: the vector of new values, for the f* prediction
    :param poly: determines which feature space to be used

    :return: The function returns the posterior distribution of the "w" weights, the f* prediction distribution, and
    the standard deviation of this distribution
    """

    A = (Phi(x_obs, N, poly) @ Phi(x_obs, N, poly).T) / np.square(sigma) + np.linalg.inv(np.eye(N))
    inv_A = np.linalg.inv(A)
    w_bar = inv_A.T @ Phi(x_obs, N, poly) @ y_obs / np.square(sigma)

    # the posterior is just a Gaussian distribution
    posterior_w = np.random.multivariate_normal(w_bar, inv_A)

    # the distribution of the prediction f*
    prediction_mean = Phi(x_star, N, poly).T @ w_bar
    prediction_var = Phi(x_star, N, poly).T @ inv_A @ Phi(x_star, N, poly)
    st_dev = np.sqrt(np.diag(prediction_var))
    prediction_distr = np.random.multivariate_normal(prediction_mean, prediction_var)

    return posterior_w, prediction_distr, st_dev


x_pred = np.linspace(-2, 2, 100)

# show the results for both the polynomial and the sinusoidal feature space
bool = [True, False]
for iter in bool:
    for feature_dim in range(1, 10):
        res, y_pred, var_pred = GaussianProcess(sigma=0.05, N=feature_dim, x_obs=x_i, y_obs=y_i, x_star=x_pred, poly=iter)
        plt.plot(x_pred, y_pred, label=r"$f^{*}$")
        plt.fill_between(x_pred, y_pred-var_pred, y_pred+var_pred, alpha=0.4, label=r"Var($f^*$)")
        plt.plot(x_i, y_i, "o")
        plt.xlabel(r"$x_i$")
        plt.ylabel(r"$y_i$")
        plt.legend(loc='best')
        if iter is True:
            plt.title(f"Feature space of dimension {feature_dim - 1}. Polynomial feature space")
        else:
            plt.title(f"Feature space of dimension {feature_dim - 1}. Sinusoidal feature space")
        plt.show()



import numpy as np
import matplotlib.pyplot as plt


def kernel(
    x, xprime, amplitude_sigma=1, correlation_length_l=1, period_T=0.5):
    exp_arg = -2/correlation_length_l**2 * \
        (np.sin(np.pi*np.abs(x-xprime)/period_T))**2
    return amplitude_sigma**2 * np.exp(exp_arg) 


if __name__ == "__main__":
    # Kernel contour plot
    xs = np.linspace(0, 1)
    ys = np.linspace(0, 1)
    X, Y = np.meshgrid(xs, ys) 
    Z = kernel(X, Y)
    plt.contourf(X, Y, Z, levels=20) 
    plt.title(r"Kernel Function Contour on $[0, 1]^2$")
    plt.draw()    

    plt.show()

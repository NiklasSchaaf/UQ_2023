import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def KernelFunction(x, x_prime, sigma=1, l=1, T=0.5):
    exponential = np.exp((-2 / l ** 2) * np.sin(np.pi * np.abs(x - x_prime) / T) ** 2)
    return sigma ** 2 * exponential


x_i = np.arange(0, 1.2, 0.2)
y_i = np.array([0, 0.59, -0.95, 0.95, -0.59, 0])

plt.plot(x_i, y_i, '-o')
plt.title("Data Observations")
plt.xlabel("r$x_i$")
plt.ylabel("r$y_i$")
plt.show()

# Make a contour plot of the kernel
NumPoints = 100
x_contour = np.linspace(0, 1, NumPoints)
KernelMat = np.zeros((NumPoints, NumPoints))

for i, x in enumerate(x_contour):
    for j, x_prime in enumerate(x_contour):
        KernelMat[i, j] = KernelFunction(x, x_prime)

plt.contourf(x_contour, x_contour, KernelMat)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("x'")
plt.title(r"Chosen Kernel $k(x, x')$")
plt.show()

test_funcs = 4
fig, axes = plt.subplots(nrows=test_funcs, ncols=1, figsize=(8, 2 * test_funcs))
for i in range(test_funcs):
    f = np.random.multivariate_normal(mean=np.zeros(NumPoints), cov=KernelMat)
    axes[i].plot(x_contour, f)
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")
    axes[i].set_title(f"Random Sample from prior # {i + 1}")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


def Kernel(x, x_prime):
    num_points_x = len(x)
    num_points_prime = len(x_prime)
    kernelmat = np.zeros((num_points_x, num_points_prime))

    for i, val in enumerate(x):
        for j, val_prime in enumerate(x_prime):
            kernelmat[i, j] = KernelFunction(val, val_prime)

    return kernelmat


def GaussianProcess(x_test, observations, sigma_nugget):
    x_obs = observations[0]
    y_obs = observations[1]

    dim = len(x_obs)
    K_inv_reg = np.linalg.inv(Kernel(x_obs, x_obs) + np.square(sigma_nugget) * np.eye(dim))

    mean = Kernel(x_test, x_obs) @ K_inv_reg @ y_obs
    cov = Kernel(x_test, x_test) - Kernel(x_test, x_obs) @ K_inv_reg @ Kernel(x_obs, x_test)

    return mean, cov


# Draw 1000 random samples from the posteriro distribution
NumSamples = 1000
MeanGP, CovGP = GaussianProcess(x_contour, [x_i, y_i], sigma_nugget=0.01)
plt.contourf(x_contour, x_contour, CovGP)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("x'")
plt.title(r"Posterior Covariance $K(x, x')$")
plt.show()

pbar = tqdm(total=NumSamples)
for _ in range(NumSamples):
    sample_f = np.random.multivariate_normal(mean=MeanGP, cov=CovGP)
    plt.plot(x_contour, sample_f, alpha=0.3)
    pbar.update()
pbar.close()
plt.plot(x_i, y_i, 'o', color='black', markersize=7, label="Observations")
plt.plot(x_contour, MeanGP, color='black', label="Mean GP")
plt.legend(loc='best')
plt.title(f"{NumSamples} Samples from GP posterior")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


def CholeskyGP(x_test, observations, sigma_nugget):
    x_obs = observations[0]
    y_obs = observations[1]

    dim = len(x_obs)
    L = np.linalg.cholesky(Kernel(x_obs, x_obs) + np.square(sigma_nugget) * np.eye(dim))
    first_op = np.linalg.solve(L, y_obs)
    alpha = np.linalg.solve(L.T, first_op)

    points_num = len(x_test)
    PointMeans = np.zeros(points_num)
    PointVars = np.zeros(points_num)

    for i, val in enumerate(x_test):
        k = Kernel(x_obs, np.array([val]))
        PointMeans[i] = k.T @ alpha
        v = np.linalg.solve(L, k)
        PointVars[i] = Kernel(np.array([val]), np.array([val])) - v.T @ v

    return PointMeans, PointVars


means, vars = CholeskyGP(x_contour, [x_i, y_i], sigma_nugget=0.01)
stdev =  np.sqrt(vars)

fig, axs = plt.subplots(2,1)
axs[0].plot(x_contour, means, label='Mean Cholesky GP')
axs[0].fill_between(x_contour, means-2*stdev, means+2*stdev, alpha=0.3, label = '95% CI')
axs[0].scatter(x_i, y_i, label='data', color = 'C0')
axs[0].legend()
axs[0].set_title('GP with Cholesky decomposition')
axs[0].set_ylabel('y')

axs[1].plot(x_contour, MeanGP, label='Mean Without Cholesky')
st_dev = np.sqrt(np.diag(CovGP))
axs[1].fill_between(x_contour, means-2*st_dev, means+2*st_dev, alpha = 0.3, label='95% CI')
axs[1].scatter(x_i, y_i, label='data', color='C0')
axs[1].legend()
axs[1].set_title('GP without Cholesky decomposition')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.tight_layout()
plt.show()
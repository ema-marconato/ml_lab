import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_decision_surface(X, Y, clf):
    """Print the decision surface of a trained sklearn classifier"""

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    colors = ["orange" if y == 1 else "blue" for y in Y]

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
    ax.scatter(X0, X1, c=colors, cmap=plt.cm.coolwarm, s=20)
    ax.set_ylabel('$x_0$')
    ax.set_xlabel('$x_1$')
    ax.set_title('Decision surface of the classifier')
    plt.show()


def plot_3D_decision_surface(X, Y, clf):
    """Plot the decision surface of a trained sklearn classifier"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Separate positive examples from negative ones
    X_positive = X[Y == 1]
    X_negative = X[Y == 0]

    ax.scatter(X_positive[:, 0], X_positive[:, 1],
               X_positive[:, 2], color="orange")
    ax.scatter(X_negative[:, 0], X_negative[:, 1],
               X_negative[:, 2], color="blue")

    # Constructing a hyperplane using a formula.
    x_points = np.linspace(-0.5, 0.5)
    y_points = np.linspace(-0.5, 0.5)
    x_points, y_points = np.meshgrid(x_points, y_points)

    w = clf.coef_[0]           # w consists of 2 elements
    b = clf.intercept_[0]      # b consists of 1 element
    z_points = -(w[0]/w[2])*x_points - (w[1]/w[2])*y_points - b/w[2]

    ax.plot_surface(x_points, y_points, z_points, color="red", alpha=0.5)
    plt.ylim(0, 3)
    plt.xlim(0, 3)
    ax.set_zlim(0, 3)
    ax.view_init(0, -90)
    plt.tight_layout()
    plt.show()


def plot_svm_margin(X_train, Y, svc_model):

    # Separate positive examples from negative ones
    X_positive = X_train[Y == 1]
    X_negative = X_train[Y == 0]

    plt.scatter(X_positive[:, 0], X_positive[:, 1],
                c="orange", label="Positive")
    plt.scatter(X_negative[:, 0], X_negative[:, 1], c="blue", label="Negative")

    # Constructing a hyperplane using a formula.
    x_points = np.linspace(-10, 10)
    w = svc_model.coef_[0]           # w consists of 2 elements
    b = svc_model.intercept_[0]      # b consists of 1 element
    y_points = -(w[0]/w[1])*x_points - b/w[1]  # getting corresponding y-points
    # Plotting a red hyperplane
    plt.plot(x_points, y_points, c='r')
    # Step 2 (unit-vector):
    w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))
    # Step 3 (margin):
    margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))
    # Step 4 (calculate points of the margin lines):
    decision_boundary_points = np.array(list(zip(x_points, y_points)))
    points_of_line_above = decision_boundary_points + w_hat * margin
    points_of_line_below = decision_boundary_points - w_hat * margin
    # Plot margin lines
    # Blue margin line above
    plt.plot(points_of_line_above[:, 0],
             points_of_line_above[:, 1],
             'y--',
             linewidth=2)
    # Green margin line below
    plt.plot(points_of_line_below[:, 0],
             points_of_line_below[:, 1],
             'g--',
             linewidth=2)
    plt.title("SVM Margin")
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.show()


def plot_data(X, Y):
    # Separate positive examples from negative ones
    X_positive = X[Y == 1]
    X_negative = X[Y == 0]

    plt.scatter(X_positive[:, 0], X_positive[:, 1],
                c="orange", label="Positive")
    plt.scatter(X_negative[:, 0], X_negative[:, 1], c="blue", label="Negative")
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.title("Binary Classification Dataset")
    plt.legend()

    plt.show()


def plot_kmeans_decision_surface(X, pred, clf):

    def create_cluster_dict(pred, X):
        tmp = {}
        for c, x in zip(pred, X):
            tmp.update({c: tmp.get(c, [])+[x]})
        return tmp

    fig, ax = plt.subplots(figsize=(4, 4))

    X = X.to_numpy()

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    plot_contours(ax, clf, xx, yy, alpha=0.3)
    c = 0
    tmp = create_cluster_dict(pred, X)
    for i in tmp.values():
        i = np.array(i)
        plt.scatter(i[:, 0], i[:, 1], alpha=0.8)
        c = c + 1
    plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[
                :, 1], s=100, color="red")
    plt.ylabel("Longitude")
    plt.xlabel("Latitude")
    ax.set_title('Decision surface of K-means')
    plt.show()


def plot_pca_clusters(X_pca, y, dimensions=False):
    fig = plt.figure()

    def create_cluster_dict(pred, X):
        tmp = {}
        for c, x in zip(pred, X):
            tmp.update({c: tmp.get(c, [])+[x]})
        return tmp

    clusters = create_cluster_dict(y, X_pca)
    if dimensions:
        ax = fig.add_subplot(projection="3d")
    c = 1
    for k, i in clusters.items():
        i = np.array(i)
        if dimensions:
            ax.scatter(i[:, 0], i[:, 1], i[:, 2], label=f"Wine Type {k}")
        else:
            plt.scatter(i[:, 0],i[:,1],label=f"Wine Type {k}")
    plt.legend()
    plt.show()


def generate_data(mu, sigma, N=2500, radius=1):
    """Generate a binary classification dataset using the formula above.

    :param mu: the mean of the normal distribution
    :param sigma: the standard deviation of the normal distribution
    :param N: number of samples for the dataset
    :param radius: radius of the circle
    """
    np.random.seed(42)
    X = []
    Y = []
    while (True):

        x = np.random.normal(mu, sigma, [2])

        if len(X) < N//2:
            if (x[0]**2 + x[1]**2) > radius**2:
                X.append(x)
                Y.append(0)
        else:
            if 0 <= (x[0]**2 + x[1]**2) <= (radius/2)**2:
                X.append(x)
                Y.append(1)

        if len(X) == N:
            break

    return np.array(X), np.array(Y)

def generate_data_bivariate(mu, sigma, N=2500, radius=1):
    """Generate a binary classification dataset using the formula above.

    :param mu: the mean of the normal distribution
    :param sigma: the standard deviation of the normal distribution
    :param N: number of samples for the dataset
    :param radius: radius of the circle
    """
    np.random.seed(42)
    X = []
    Y = []
    while (True):
        sw = np.random.randint(0,1)
        if sw == 0:	  mu_s = -mu
        else:		  mu_s = mu
	
        x = np.random.normal(mu_s, sigma, [2])

        if (x[0]**2 + x[1]**2 - mu_s**2) > radius**2:
                X.append(x)
                Y.append(0)
        else:
            if 0 <= (x[0]**2 + x[1]**2 - mu_s**2) <= (radius/2)**2:
                X.append(x)
                Y.append(1)

        if len(X) == N:
            break

    return np.array(X), np.array(Y)
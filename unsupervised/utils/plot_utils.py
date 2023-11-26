import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Ellipse
from sklearn import datasets


def plot_blobs(X, labels_true):
    fig = plt.figure(figsize=(5,5))

    colors = ["#4EACC5", "#FF9C34", "#4E9A06", 'red']
    for i, label in enumerate(np.unique(labels_true)):
        X_label = X[labels_true == label]
        X1, X2 = np.split(X_label, 2, axis=1)
        plt.scatter(X1, X2, s=1.5, color=colors[i], label=label, marker='.')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

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


def plot_kmeans_decision_boundaries(X, kmeans_model):
    """
    Plot the decision boundaries (contours) of a KMeans model.

    Parameters:
    - X (numpy array): Input data for clustering.
    - kmeans_model: Fitted KMeans model.
    """
    # Step size of the meshgrid
    h = 0.02

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get predicted labels for each point in the meshgrid
    Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels_, cmap=plt.cm.rainbow, edgecolor='k', s=40)
    
    # Plot the cluster centers
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='black')

    plt.title("KMeans Decision Boundaries")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plot an ellipse for a given covariance matrix and position.

    Parameters:
        - cov: Covariance matrix.
        - pos: Position (mean) of the ellipse.
        - nstd: The radius of the ellipse in terms of standard deviations.
        - ax: Matplotlib axes on which to plot the ellipse.
        - **kwargs: Additional keyword arguments for matplotlib.patches.Ellipse.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate ellipse parameters
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Calculate ellipse angle and center
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi
    center = pos

    # Create and plot the ellipse
    ell = Ellipse(xy=center, width=v[0] * nstd, height=v[1] * nstd, angle=angle, **kwargs)
    ax.add_patch(ell)

    return ell

def make_ellipses(gmm, ax):
    colors = ["navy", "turquoise", "darkorange"]

    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")


def plot_K_means(X, n_clusters, 
                 k_means, k_means_labels, k_means_cluster_centers, 
                 mbk, mbk_means_labels, mbk_means_cluster_centers, 
                 t_batch, t_mini_batch):
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.3fs\ninertia: %f" % (t_batch, k_means.inertia_))

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("MiniBatchKMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, "train time: %.3fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

    # Initialize the different array to all False
    different = mbk_means_labels == 4
    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
        different += (k_means_labels == k) != (mbk_means_labels == k)

    identical = np.logical_not(different)
    ax.plot(X[identical, 0], X[identical, 1], "w", markerfacecolor="#bbbbbb", marker=".")
    ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
    ax.set_title("Difference")
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()

def plot_kmeans_clusters(X, labels_list, cluster_centers_list):
    """
    Plot points, clusters, and cluster centers for KMeans models.

    Parameters:
    - X (numpy array): Input data for clustering.
    - labels_list (list): List of cluster labels for each model.
    - cluster_centers_list (list): List of cluster centers for each model.
    """
    n_models = len(labels_list)
    n_clusters = 10  # Assuming labels are 0-indexed
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    num_rows = 2
    num_cols = (n_models + 1) // num_rows  # Calculate the number of columns needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.subplots_adjust(hspace=0.5)

    for i, (labels, cluster_centers) in enumerate(zip(labels_list, cluster_centers_list), 1):
        # Scatter plot of data points with colors based on assigned labels
        ax = axes[(i - 1) // num_cols, (i - 1) % num_cols]

        n_clusters = len(cluster_centers)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            ax.scatter(X[my_members, 0], X[my_members, 1], c=[col], label=f'Cluster {k}', s=1)

        ax.set_title(f"Number of clusters {i}")
        ax.set_xticks(())
        ax.set_yticks(())
        # ax.legend(ncol=2)
        # plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=2)
    

    # Hide any unused subplots
    for j in range(i, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.show()


def estimate_GMMS(estimators, X_train, X_test, y_train, y_test):
    n_estimators = len(estimators)
    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(
        bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
    )

    colors = ["navy", "turquoise", "darkorange"]

    iris = datasets.load_iris()

    for index, (name, estimator) in enumerate(estimators.items()):   

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)

        for n, color in enumerate(colors):
            data = iris.data[iris.target == n]
            plt.scatter(
                data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n]
            )
        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

        y_train_pred = estimator.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

        y_test_pred = estimator.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))
    plt.show()
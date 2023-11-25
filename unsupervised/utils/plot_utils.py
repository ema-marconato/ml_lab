import matplotlib.pyplot as plt
import numpy as np

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



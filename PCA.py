import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_PCA(X, n_components=4):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)

    # Convert the result to a DataFrame
    pca_df = pd.DataFrame(pca_result)

    # # Plot cumulative explained variance
    # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # plt.figure(figsize=(8,5))
    # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    # plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance threshold')
    # plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Cumulative Explained Variance')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # Find the smallest n_components that explain at least 95% variance
    # optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
    # print(f"Optimal number of components: {optimal_components}")

    # Return the transformed data as a DataFrame and explained variance
    return pca, pca_df, pca.explained_variance_ratio_

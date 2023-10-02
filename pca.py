import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from visualize import visualize_pca


def compute_pca(x, n, y):
    if n is None:
        pca = PCA()

        pca.fit(x)
        plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
        plt.title('Cumulative explained variance by number of principal components')
        plt.show()

        loadings = pd.DataFrame(
            data=pca.components_.T * np.sqrt(pca.explained_variance_),
            columns=[f'PC{i}' for i in range(1, len(x.columns) + 1)],
            index=x.columns
        )
        print(loadings)

        pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
        pc1_loadings = pc1_loadings.reset_index()
        pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

        plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
        plt.title('PCA loading scores (first principal component)')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.show()
        #
    else:

        pca = PCA(n_components=n)

        principalComponents = pca.fit_transform(x)
        columns = [f'principal component {i+1}' for i in range(n)]
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = columns)

        THRESHOLD = 50
        targetDf = pd.DataFrame(data=(y > THRESHOLD), columns=['is_good'])
        finalDf = pd.concat([principalDf, targetDf], axis = 1)

        visualize_pca(finalDf, f'PCA, t:{THRESHOLD}%', [True, False], ['r', 'b'], 'is_good')





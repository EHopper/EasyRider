import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.manifold


def fit_tsne_2d(df):
    mapped_clusters = sklearn.manifold.TSNE(
                        n_components=2, metric='cosine', init='pca'
                      ).fit_transform(df)

    return mapped_clusters[:, 0], mapped_clusters[:, 1]

def plot_cluster_2d(df, labels, x, y):
    # x, y = fit_tsne_2d(df)
    n_clusts = len(set(labels))
    colours = sns.color_palette('husl', n_clusts - min(labels))
    colours = [(0.8, 0.8, 0.8)] + colours # Plot cluster -1 (unclustered data) as grey
    plt.figure(figsize=(10, 4))
    for c in range(min(labels), n_clusts - min(labels)):
        plt.plot(x[labels == c], y[labels == c], '.', color=colours[c + 1])
        if c > -1:
            plt.text(np.mean(x[labels == c]),
                     np.mean(y[labels == c]), '{}'.format(c), )

def plot_hists(df):

    features = df.columns.tolist()
    features.remove('labels')
    nf = len(features)

    plt.figure(figsize=(15, 5))
    for i, f in enumerate(features):
        plt.subplot(1, nf, i + 1)
        plt.hist(df[f])

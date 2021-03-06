import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def get_data(self):
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        return data

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()


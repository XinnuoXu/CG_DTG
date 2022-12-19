#coding=utf8

import json
import numpy as np
from sklearn.manifold import TSNE

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

'''
def 3d_plot(filename):
    for i, line in enumerate(open(filename)):
        json_obj = json.loads(line.strip())
        inputs = json_obj['inputs'][0]
        slots = json_obj['slots'][0]
        X = np.array(inputs+slots)
        X_embedded = TSNE(n_components=3, init='random', perplexity=30).fit_transform(X)

        ninputs = len(inputs)
        ax = plt.axes(projection='3d')
        x = X_embedded[:ninputs,0]
        y = X_embedded[:ninputs,1]
        z = X_embedded[:ninputs,2]
        ax.scatter(x, y, z, c=z, cmap='viridis', marker='^', linewidth=0.5);
        x = X_embedded[ninputs:,0]
        y = X_embedded[ninputs:,1]
        z = X_embedded[ninputs:,2]
        ax.scatter(x, y, z, c=z, cmap='gray', marker='o', linewidth=0.5);
        plt.savefig(f'emb_{i}.png')

def 2d_plot(filename):
    for i, line in enumerate(open(filename)):
        json_obj = json.loads(line.strip())
        inputs = json_obj['inputs'][0]
        slots = json_obj['slots'][0]
        X = np.array(inputs+slots)
        X_embedded = TSNE(n_components=2, init='random', perplexity=30).fit_transform(X)

        ninputs = len(inputs)
        x = X_embedded[:ninputs,0]
        y = X_embedded[:ninputs,1]
        plt.scatter(x, y, c='b', marker='^', linewidth=0.5, alpha=0.5);
        x = X_embedded[ninputs:,0]
        y = X_embedded[ninputs:,1]
        ax.scatter(x, y, c='r', marker='o', linewidth=0.5, alpha=0.5);
        plt.savefig(f'emb_{i}.png')
'''

if __name__ == '__main__':
    for i, line in enumerate(open('./111')):
        json_obj = json.loads(line.strip())
        inputs = json_obj['inputs'][0]
        slots = json_obj['slots'][0]
        X = np.array(inputs+slots)
        X_embedded = TSNE(n_components=2, init='random', perplexity=30).fit_transform(X)

        ninputs = len(inputs)
        plt.clf()
        x = X_embedded[:ninputs,0]
        y = X_embedded[:ninputs,1]
        plt.scatter(x, y, c='b', marker='^', linewidth=0.5, alpha=0.5);
        x = X_embedded[ninputs:,0]
        y = X_embedded[ninputs:,1]
        plt.scatter(x, y, c='r', marker='o', linewidth=0.5, alpha=0.5);
        plt.savefig(f'emb_{i}.png')

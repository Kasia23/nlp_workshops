import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def create_matrix_from_labels(labels_list, names):
    matrix = np.zeros((len(labels_list), len(names)))
    for i,entry in enumerate(labels_list):
        for category in entry:
            if category in names:
                matrix[i, names.index(category)] = 1
#         if len(entry)==1:
#             matrix[i, len(names)-1] = 1
    return matrix

def plot_matrix(a, names, title=''):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(a)
    ax.set_xticklabels(['']+names, rotation=90)
    ax.set_yticklabels(['']+names)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title(title, y=-0.05)
    fig.colorbar(cax)
    return
    
def cooccurrence_matrix(matrix):
    return np.divide(np.dot(matrix.T, matrix),np.diagonal(np.dot(matrix.T, matrix))).T


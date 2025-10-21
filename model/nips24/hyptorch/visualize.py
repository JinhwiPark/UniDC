from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import numpy as np

def vis_pca(x):

    # ipca = IncrementalPCA(n_components=1)
    ipca_3ch = IncrementalPCA(n_components=3)

    affinity_pca = (x[0].permute(1, 2, 0)).cpu().detach().numpy()
    H, W, C = affinity_pca.shape
    affinity_pca = affinity_pca.reshape(-1, C)
    ipca_3ch.fit(affinity_pca)
    affinity_pca = ipca_3ch.transform(affinity_pca)
    affinity_pca = affinity_pca.reshape(H, W, 3)
    affinity_pca_max = affinity_pca.max()
    affinity_pca_min = affinity_pca.min()
    affinity_pca = (affinity_pca - affinity_pca_min) / (affinity_pca_max - affinity_pca_min)
    affinity_pca = np.uint8(affinity_pca * 255)
    plt.imshow(affinity_pca, aspect='auto', cmap='hsv')
#
# def vis_pca(x):
#
#     # ipca = IncrementalPCA(n_components=1)
#     ipca_3ch = IncrementalPCA(n_components=3)
#
#     affinity_pca = (x[0].permute(1, 2, 0)).cpu().detach().numpy()
#     H, W, C = affinity_pca.shape
#     affinity_pca = affinity_pca.reshape(-1, C)
#     ipca_3ch.fit(affinity_pca)
#     affinity_pca = ipca_3ch.transform(affinity_pca)
#     affinity_pca = affinity_pca.reshape(H, W, 3)
#     affinity_pca_max = affinity_pca.max()
#     affinity_pca_min = affinity_pca.min()
#     affinity_pca = (affinity_pca - affinity_pca_min) / (affinity_pca_max - affinity_pca_min)
#     affinity_pca = np.uint8(affinity_pca * 255)
#     plt.imshow(affinity_pca, aspect='auto', cmap='hsv')
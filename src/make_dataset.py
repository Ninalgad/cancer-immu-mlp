import numpy as np
import scanpy as sc
import tensorflow as tf
from sklearn.decomposition import PCA


LABEL_ENCODING = {
    'progenitor': 0,
    'effector': 1,
    'terminal exhausted': 2,
    'cycling': 3,
    'other': 4
}


def process_adata(adata, g2v_embeddings):
    Q = []
    for p in adata.obs['condition']:
        p = p.lower()
        if p in g2v_embeddings:
            q = g2v_embeddings[p]
        else:
            q = np.zeros((200,), 'float32')
        Q.append(q)
    Q = np.array(Q)

    X = adata.X.toarray()
    Y = [LABEL_ENCODING[y] for y in adata.obs['state']]
    Y = tf.keras.utils.to_categorical(Y, num_classes=max(Y) + 1)
    Y = np.array(Y)

    return Q, X, Y


def make_dataset(adata_path, g2v_embeddings):

    adata = sc.read_h5ad(adata_path)
    Q, X, Y = process_adata(adata, g2v_embeddings)
    conditions = adata.obs['condition']

    # reduce the dimensionality
    pca = PCA(n_components=128)
    X = pca.fit_transform(X)

    return Q, X, Y, conditions


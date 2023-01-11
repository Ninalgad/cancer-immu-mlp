import numpy as np


def load_embeddings(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]

    g2v_embeddings = {}
    for x in lines:
        x = x.split(' ')
        g = x[0].lower()
        g2v_embeddings[g] = np.asarray(x[1:-1], 'float32')

    return g2v_embeddings


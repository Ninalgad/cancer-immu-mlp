import numpy as np
import pickle


EMB_PATH = "./data/embeddings/g2v.pkl"


def load_embeddings():

    with open(EMB_PATH, 'rb') as f:
        g2v_embeddings = pickle.load(f)

    for g in g2v_embeddings:
        g2v_embeddings[g] = np.array(g2v_embeddings[g], 'float32')

    return g2v_embeddings


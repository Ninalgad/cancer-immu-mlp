from loguru import logger
from pathlib import Path
import typer
import numpy as np

from src.make_dataset import make_dataset
from src.g2v import load_embeddings
from src.training_utils import train_model


def validation_subset(conditions, labels, size=8):
    subset = []
    # refine based on number of samples of the perturbation
    v = np.array(conditions)
    v, c = np.unique(v, return_counts=True)
    q25, q75 = np.quantile(c, 0.25), np.quantile(c, 0.75)
    for gene, count in zip(v, c):
        if (q25 < count) and (count < q75):
            subset.append(gene)

    # refine based on distance from the mean proportions
    dist = []
    mean = np.array((0.0675, 0.2097, 0.3134, 0.3921, 0.0173), 'float32')
    for p in subset:
        y = labels[conditions == p].sum(0)
        y = y / y.sum()
        # d = np.abs(y - mean).sum() # l1
        d = -(mean * np.log(y + 1e-10)).sum()  # neg loglik
        dist.append(d)

    # select the 'size' farthest number of genes
    idx = np.argsort(dist)[-size:]
    subset = [subset[i] for i in idx]
    return subset


def batch(iterable, n=1):
    size = len(iterable)
    for ndx in range(0, size, n):
        yield iterable[ndx:min(ndx + n, size)]


def main(
        model_dir: Path = typer.Option(
            "./data/processed", help="Directory to save the output model weights in npy format"
        ),
        features_dir: Path = typer.Option(
            "./data/raw/", help="Path to the raw features"
        ),
        embedding_path: Path = typer.Option(
            "./data/embeddings/gene2vec_dim_200_iter_9_w2v.txt", help="Path to the Gene2Vec embeddings"
        ),
        n_folds: int = typer.Option(
            8, help="Number of folds/models, Must be at least 2"
        ),
        n_epochs: int = typer.Option(
            50, help="Number of training epochs, Must be at least 1"
        ),
        random_state: int = typer.Option(
            758625225, help="Controls the randomness of each fold and noise"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    np.random.seed(random_state)

    if debug:
        logger.info("Running in debug mode")
        n_epochs = 1

    logger.info(f"Loading embeddings from {embedding_path}")
    g2v_embeddings = load_embeddings(embedding_path)

    logger.info(f"Creating dataset from {features_dir}")
    Q, X, Y, conditions = make_dataset(
        features_dir / 'sc_training.h5ad', g2v_embeddings
    )
    unique_perturb = np.array(sorted(conditions.unique()))
    val_perturbations = validation_subset(conditions, Y)
    np.random.shuffle(val_perturbations)

    test_batch_size = max(1, len(val_perturbations) // n_folds)
    test_perturb_gen = batch(val_perturbations, test_batch_size)
    logger.info(f"Training {n_folds} models")
    for (i, perturb_test) in enumerate(test_perturb_gen):
        perturb_train = [p for p in unique_perturb if p not in perturb_test]

        train_index = np.array([p in perturb_train for p in conditions])
        test_index = np.logical_not(train_index)

        q_train, q_test = Q[train_index], Q[test_index]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        f = model_dir / f'model-{i}'
        val_score = train_model(x_train, q_train, y_train,
                                y_test, perturb_test, conditions[test_index],
                                g2v_embeddings,
                                h5_name=f, batch_size=128, epochs=n_epochs)
        logger.info(f"Trained model {i}, Validation metric: {val_score}")

    logger.success(f"Completed Training models to {model_dir}")


if __name__ == "__main__":
    typer.run(main)

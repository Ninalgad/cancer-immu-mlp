from loguru import logger
from pathlib import Path
import typer
from sklearn.model_selection import KFold
import numpy as np

from src.make_dataset import make_dataset
from src.g2v import load_embeddings
from src.training_utils import train_model


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
            10, help="Number of folds, Must be at least 2"
        ),
        random_state: int = typer.Option(
            758625225, help="Controls the randomness of each fold"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    n_epochs = 50
    if debug:
        logger.info("Running in debug mode")
        n_epochs = 1
        n_folds = 2

    logger.info(f"Loading embeddings from {embedding_path}")
    g2v_embeddings = load_embeddings(embedding_path)

    logger.info(f"Creating dataset from {features_dir}")
    Q, X, Y, conditions = make_dataset(
        features_dir / 'sc_training.h5ad', g2v_embeddings
    )
    unique_perturbations = np.array(sorted(conditions.unique()))

    logger.info("Training models")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(unique_perturbations)):
        perturb_train = unique_perturbations[train_index]
        perturb_test = unique_perturbations[test_index]

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

    logger.info(f"Completed Training models to {model_dir}")


if __name__ == "__main__":
    typer.run(main)


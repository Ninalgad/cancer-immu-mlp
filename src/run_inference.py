from loguru import logger
import pandas as pd
from pathlib import Path
import typer
import tensorflow as tf
import numpy as np

from src.g2v import load_embeddings
from src.model import SimpNet
from src.eval import sample_dist
from make_dataset import LABEL_ENCODING


def main(
        test_mode: bool = typer.Option(
            False, help="Predict the test genes, otherwise predict the validation genes"
        ),
        submission_save_dir: Path = typer.Option(
            "./data/processed", help="Predict the test genes, otherwise predict the validation genes"
        ),
        model_dir: Path = typer.Option(
            "./data/processed", help="Directory to save the output model weights"
        ),
        embedding_path: Path = typer.Option(
            "./data/embeddings/gene2vec_dim_200_iter_9_w2v.txt", help="Path to the Gene2Vec embeddings"
        ),
        n_samples: int = typer.Option(
            1000, help="Path to the Gene2Vec embeddings"
        )
):
    genes = ['Aqr', 'Bach2', 'Bhlhe40']
    submission_name = "validation_output.csv"
    if test_mode:
        genes = ['Ets1', 'Fosb', 'Mafk', 'Stat3']
        submission_name = "test_output.csv"
    genes = [g.lower() for g in genes]

    logger.info(f"Loading embeddings from {embedding_path}")
    g2v_embeddings = load_embeddings(embedding_path)

    logger.info("Creating model")
    tf.keras.backend.clear_session()
    model = SimpNet(128 + 64)

    @tf.function
    def predict(q, z):
        return model.call(q, z)

    logger.info("Predicting labels")
    predictions = np.zeros((len(genes), 5), 'float32')
    for i in range(10):
        h5_path = model_dir / f"model-{i}.h5"
        model.load_weights(h5_path)
        for j, g in enumerate(genes):
            dist = sample_dist(predict, g2v_embeddings[g], n_samples)
            dist = dist / len(genes)
            predictions[i] = predictions[i] + dist

    # normalise the predictions
    for i in range(len(genes)):
        predictions[i] = predictions[i] / predictions[i].sum()

    # generate submission
    submission = {"genes": genes}
    for i in LABEL_ENCODING:
        submission[LABEL_ENCODING[i]] = predictions[:, i]
    submission = pd.DataFrame(submission)

    submission.to_csv(submission_save_dir / submission_name, index=False)
    logger.success(f"Submission saved to {submission_save_dir / submission_name}")


if __name__ == "__main__":
    typer.run(main)


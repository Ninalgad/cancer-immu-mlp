from loguru import logger
from pathlib import Path
import typer
import numpy as np

from src.make_dataset import make_dataset
from src.g2v import load_embeddings
from src.training_utils import train_model, run_knowledge_distillation
from src.validation_subset import validation_subset


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
        n_folds: int = typer.Option(
            7, help="Number of folds/models, Must be at least 2"
        ),
        teacher_dim: int = typer.Option(
            128, help="The hidden dimension of the teacher (NOT used for inference) MLP"
        ),
        student_dim: int = typer.Option(
            256, help="The hidden dimension of the student (used for inference) MLP"
        ),
        random_state: int = typer.Option(
            758625225, help="Controls the randomness of each fold and noise"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    np.random.seed(random_state)

    n_kd_epochs = 75
    n_epochs = 50
    if debug:
        logger.info("Running in debug mode")
        n_epochs = 1
        n_kd_epochs = 1

    logger.info(f"Loading embeddings")
    g2v_embeddings = load_embeddings()

    logger.info(f"Creating dataset from {features_dir}")
    Q, X, Y, conditions = make_dataset(
        features_dir / 'sc_training.h5ad', g2v_embeddings
    )
    unique_perturb = np.array(sorted(conditions.unique()))
    val_perturbations = validation_subset(conditions, Y)
    np.random.shuffle(val_perturbations)

    # training loop
    test_batch_size = max(1, len(val_perturbations) // n_folds)
    test_perturb_gen = batch(val_perturbations, test_batch_size)
    scores, teacher_h5s = [], []
    logger.info(f"Training {n_folds} models")
    for (i, perturb_test) in enumerate(test_perturb_gen):
        perturb_train = [p for p in unique_perturb if p not in perturb_test]
        train_index = np.array([p in perturb_train for p in conditions])
        test_index = np.logical_not(train_index)

        q_train, q_test = Q[train_index], Q[test_index]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        f = model_dir / f'teacher-model-{i}'
        val_score = train_model(x_train, q_train, y_train,
                                y_test, perturb_test, conditions[test_index],
                                g2v_embeddings,
                                h5_name=f, batch_size=128, epochs=n_epochs,
                                learning_rate=1e-4, model_dim=teacher_dim)
        teacher_h5s.append(f)
        scores.append(val_score)
        logger.info(f"Trained model {i}, Validation metric: {val_score}")

    logger.info(f"Average Teacher Validation metric: {np.mean(scores)}")

    # knowledge distillation learning from generated data/labels from teachers
    logger.info(f"Performing knowledge-distillation for {n_kd_epochs} epochs")
    ssl_h5 = model_dir / f'distilled-model'
    run_knowledge_distillation(ssl_h5, X.shape[-1], teacher_h5s,
                               val_perturbations, g2v_embeddings,
                               teacher_model_dim=teacher_dim, model_dim=student_dim,
                               epochs=n_kd_epochs, batch_size=1042)

    # fine-tune on labeled data
    scores = []
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
                                h5_name=f, batch_size=128, epochs=n_epochs,
                                learning_rate=1e-4, model_dim=student_dim,
                                pretrained_h5_path=str(ssl_h5) + '.h5')
        scores.append(val_score)
        logger.info(f"Trained model {i}, Validation metric: {val_score}")

    logger.info(f"Average Final Validation metric: {np.mean(scores)}")
    logger.success(f"Completed Training Inference Models to {model_dir}")


if __name__ == "__main__":
    typer.run(main)

import numpy as np
import tensorflow as tf
import gc

from src.model import SimpNet
from src.loss import loss_function
from src.eval import evaluate


def create_datagen(x, q, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices({'q': q, 'x': x, 'y': y})
    dataset = dataset.repeat().shuffle(100_000).batch(batch_size)
    return dataset.as_numpy_iterator()


def train_model(x_train, q_train, y_train,
                y_test, perturbations_test, condition_test, g2v_embeddings,
                h5_name='temp', batch_size=128, epochs=50):
    tf.keras.backend.clear_session()
    model = SimpNet(x_train.shape[-1])
    optimizer = tf.keras.optimizers.Adam(1e-4, epsilon=1e-08)

    @tf.function
    def train_step(q, z, x, y):
        with tf.GradientTape() as tape:
            pred_y, pred_x = model.call(q, z)
            loss = loss_function(y, pred_y, x, pred_x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def predict(q, z):
        return model.call(q, z)

    train_gen = create_datagen(x_train, q_train, y_train, batch_size)

    # training loop
    steps_per_epoch = len(x_train) // batch_size
    best_val = np.inf
    for epoch in range(0, epochs):
        for _ in range(steps_per_epoch):
            batch = next(train_gen)
            batch['z'] = np.random.normal(scale=.5, size=(batch_size, 64))
            batch['z'] = batch['z'].astype('float32')
            train_step(**batch)
        val = evaluate(predict, perturbations_test, condition_test,
                       y_test, g2v_embeddings)
        gc.collect()
        if val < best_val:
            model.save_weights(str(h5_name) + '.h5')
            best_val = val

    return best_val


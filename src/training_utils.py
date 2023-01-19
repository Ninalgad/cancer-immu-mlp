import numpy as np
import tensorflow as tf
import gc
import tqdm

from src.model import SimpNet
from src.loss import loss_function
from src.eval import evaluate


def make_ds(x, q, y):
    ds = tf.data.Dataset.from_tensor_slices({'q': q, 'x': x, 'y': y})
    ds = ds.shuffle(100_000).repeat()
    return ds


def create_datagen(x, q, y, batch_size):
    label = np.argmax(y, -1)
    class_ds = []
    for i in range(label.max() + 1):
        idx = label == i
        ds = make_ds(x[idx], q[idx], y[idx])
        class_ds.append(ds)

    dataset = tf.data.Dataset.sample_from_datasets(class_ds,
                                                   weights=[.2, .2, .2, .2, .2])
    dataset = dataset.batch(batch_size)
    return dataset.as_numpy_iterator()


def train_model(x_train, q_train, y_train,
                y_test, perturbations_test, condition_test, g2v_embeddings,
                h5_name='temp', batch_size=128, epochs=50,
                learning_rate=1e-4, model_dim=128, pretrained_h5_path=""):
    tf.keras.backend.clear_session()
    model = SimpNet(x_train.shape[-1], model_dim, model_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if pretrained_h5_path != "":
        # call to 'create' the model
        model(np.zeros((1, 200), 'float32'),
              np.zeros((1, 64), 'float32'))
        model.load_weights(pretrained_h5_path)

    @tf.function
    def train_step(q, z, x, y):
        with tf.GradientTape() as tape:
            pred_y, pred_x = model(q, z)
            loss = loss_function(y, pred_y, x, pred_x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def predict(q, z):
        return model(q, z)

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


def run_knowledge_distillation(h5_name, x_dim, teacher_h5s,
                               val_perturbations, g2v_embeddings,
                               teacher_model_dim=128, model_dim=256,
                               epochs=25, batch_size=128, learning_rate=1e-4):
    num_teachers = len(teacher_h5s)
    # val_perturbations = ['Elf1', 'Irf9', 'Lrp1', 'Egr1', 'Id3', 'Oxnad1', 'Foxm1']
    val_perturbations = set([g.lower() for g in val_perturbations])

    ssl_genes = [g for g in g2v_embeddings if g not in val_perturbations]

    tf.keras.backend.clear_session()
    model = SimpNet(x_dim, model_dim, model_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)
    teacher_models = [SimpNet(x_dim, teacher_model_dim, teacher_model_dim)
                      for _ in range(num_teachers)]

    @tf.function
    def train_step(q, z, x, y):
        with tf.GradientTape() as tape:
            pred_y, pred_x = model(q, z)
            loss = loss_function(y, pred_y, x, pred_x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def predict_teacher(q, z, i):
        t_model = teacher_models[i]
        return t_model(q, z)

    def sample_train_batch(batch_size):
      genes = np.random.choice(ssl_genes, size=batch_size)
      q = np.array([g2v_embeddings[x] for x in genes], 'float32')
      q = q + np.random.normal(scale=.1, size=q.shape).astype('float32')

      z = np.random.normal(scale=.5, size=(batch_size, 64)).astype('float32')
      i = np.random.choice(num_teachers)
      y, x = predict_teacher(q, z, i)

      # inp noise
      x = x + np.random.normal(scale=.1, size=x.shape).astype('float32')
      return {'q': q, 'z': z, 'x': x.numpy(), 'y': y.numpy()}

    # create teacher models
    for (t_h5, t_model) in zip(teacher_h5s, teacher_models):
        t_model(np.zeros((1, 200), 'float32'), np.zeros((1, 64), 'float32'))
        t_model.load_weights(str(t_h5) + '.h5')

    # training loop
    steps_per_epoch = len(ssl_genes) // batch_size
    for epoch in tqdm.tqdm(range(0, epochs)):
        for _ in range(steps_per_epoch):
            batch = sample_train_batch(batch_size)
            train_step(**batch)
            gc.collect()
    model.save_weights(str(h5_name) + '.h5')

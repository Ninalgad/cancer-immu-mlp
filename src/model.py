import tensorflow as tf


def encoder_three_layer(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(2 * dff),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(dff),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(d_model)
    ])


def encoder_two_layer(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(d_model)
    ])


class SimpNet(tf.keras.Model):
    def __init__(self, x_dim, q_enc_dim=128, x_env_dim=128):
        super(SimpNet, self).__init__()

        self.q_enc = encoder_three_layer(x_dim, q_enc_dim)
        self.x_enc = encoder_two_layer(5, x_env_dim)

    def call(self, q, z):
        inp = 10. * tf.concat([q, z], -1)
        x_out = self.q_enc(inp)
        p_out = tf.keras.activations.softmax(self.x_enc(x_out), -1)

        return p_out, x_out


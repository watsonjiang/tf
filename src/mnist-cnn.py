#
# mnist数据，手写识别
# NN 方法
#
import tensorflow as tf
import logging
import sys


def init_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(message)s',
        level=logging.INFO,
    )

def mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    logging.info(model.evaluate(x_test,  y_test, verbose=2))


if __name__ == "__main__":
    init_logging()
    mnist()
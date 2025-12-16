import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

if gpus:
    # Place a computation on the first GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Result of matrix multiplication on GPU:\n", c.numpy())
else:
    print("No GPU found. Running on CPU.")

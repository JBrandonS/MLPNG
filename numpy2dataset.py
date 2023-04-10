import numpy as np
import tensorflow as tf

# A helper class to save NumPy arrays to TFRecord files
# This will hopefully allow for better use of the GPU
def save_numpy_to_tf_dataset(array, filename, num_parallel_calls=tf.data.AUTOTUNE, buffer_size=None):
    # Ensure the input array has a rank of at least 1
    if array.ndim == 0:
        array = np.array([array])
        
    # Create a TensorFlow dataset from the NumPy array
    dataset = tf.data.Dataset.from_tensor_slices(array)

    # Serialize the dataset to a file
    def serialize_example(data):
        example_array = tf.io.serialize_tensor(data)
        return tf.train.Example(features=tf.train.Features(feature={
            'array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example_array.numpy()]))
        })).SerializeToString()

    def tf_serialize_example(example):
        tf_string = tf.py_function(serialize_example, (example,), tf.string)
        return tf.reshape(tf_string, ())

    # Parallelize the serialization process
    serialized_dataset = dataset.interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(x).map(tf_serialize_example, num_parallel_calls=num_parallel_calls),
        cycle_length=10,
        num_parallel_calls=num_parallel_calls
    )

    # Prefetch data for better performance
    if buffer_size is None:
        buffer_size = tf.data.AUTOTUNE

    serialized_dataset = serialized_dataset.prefetch(buffer_size)

    # Write the serialized dataset to a file
    with tf.io.TFRecordWriter(filename) as writer:
        for serialized_example in serialized_dataset:
            writer.write(serialized_example.numpy())

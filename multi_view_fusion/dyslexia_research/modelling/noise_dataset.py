import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

class RandomNoiseTensorFlowDataset:
    """
    TensorFlow-compatible dataset that generates random noise images with
    user-provided or random binary labels. The images are sized and transformed
    to be compatible with a ResNet50 model.
    """
    def __init__(self, num_images: int, image_size: int = 224, num_channels: int = 3, labels: np.ndarray = None):
        """
        Initializes the RandomNoiseTensorFlowDataset.

        Args:
            num_images (int): The total number of random images to generate.
            image_size (int): The height and width of the square images (e.g., 224 for ResNet50).
            num_channels (int): The number of color channels (e.g., 3 for RGB).
            labels (np.ndarray, optional): A 1D NumPy array of labels (0 or 1) for the images.
                                            If None, labels will be randomly generated.
        """
        if not isinstance(num_images, int) or num_images <= 0:
            raise ValueError("num_images must be a positive integer.")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if not isinstance(num_channels, int) or num_channels <= 0:
            raise ValueError("num_channels must be a positive integer.")

        self.num_images = num_images
        self.image_size = image_size
        self.num_channels = num_channels

        # Generate random noise images:
        # np.random.rand creates arrays with values uniformly sampled from [0, 1).
        self.image_arrays = np.random.rand(self.num_images, self.image_size, self.image_size, self.num_channels).astype(np.float32)

        # Assign labels 
        if labels is not None:
            if not isinstance(labels, np.ndarray):
                raise TypeError("Provided labels must be a NumPy array.")
            if labels.ndim != 1 or labels.shape[0] != num_images:
                raise ValueError(f"Provided labels array must be 1D and have {num_images} elements. Got shape {labels.shape}.")
            if not np.all((labels == 0) | (labels == 1)):
                print("Warning: Labels contain values other than 0 or 1. Ensure this is intended for your use case.")
            self.labels = labels.astype(np.int64)
        else:
            # Generate random binary labels (0 or 1):
            self.labels = np.random.randint(0, 2, size=(self.num_images,), dtype=np.int64)

        print(f"RandomNoiseTensorFlowDataset created with {num_images} images.")
        print(f"Image shape: ({image_size}, {image_size}, {num_channels})")
        print(f"Label distribution (approx): {np.sum(self.labels == 0).item()} zeros, {np.sum(self.labels == 1).item()} ones.")

    def _normalize_image(self, image):
        image = tf.cast(image, tf.float32)
        return preprocess_input(image)

    def to_tf_dataset(self, batch_size: int, shuffle: bool = True, prefetch_buffer_size: int = tf.data.AUTOTUNE):
        """
        Converts the dataset to a tf.data.Dataset.

        Args:
            batch_size (int): The batch size for the TensorFlow dataset.
            shuffle (bool): Whether to shuffle the dataset.
            prefetch_buffer_size (int): The number of batches to prefetch. Use tf.data.AUTOTUNE for automatic tuning.

        Returns:
            tf.data.Dataset: A TensorFlow dataset.
        """
        # Create a tf.data.Dataset from NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.image_arrays, self.labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_images)

        # Apply transformations (normalization) using .map()
        dataset = dataset.map(lambda image, label: (self._normalize_image(image), label),
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset
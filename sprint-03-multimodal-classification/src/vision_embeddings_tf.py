import numpy as np
import pandas as pd
import os
from transformers import TFConvNextV2Model, TFViTModel, TFSwinModel
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3
)
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.

    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size.

    Returns:
    - np.array: Preprocessed image.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0

    return img


class FoundationalCVModel:
    """
    A Keras module for loading and using foundational computer vision models.

    This class allows you to load and use various foundational computer vision models for tasks like image classification
    or feature extraction. The user can choose between evaluation mode (non-trainable model) and fine-tuning mode (trainable model).

    Attributes:
    ----------
    backbone_name : str
        The name of the foundational CV model to load (e.g., 'resnet50', 'vit_base').
    model : keras.Model
        The compiled Keras model with the selected backbone.

    Parameters:
    ----------
    backbone : str
        The name of the foundational CV model to load. The available backbones can include:
        - ResNet variants: 'resnet50', 'resnet101'
        - DenseNet variants: 'densenet121', 'densenet169'
        - InceptionV3: 'inception_v3'
        - ConvNextV2 variants: 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large'
        - Swin Transformer variants: 'swin_tiny', 'swin_small', 'swin_base'
        - Vision Transformer (ViT) variants: 'vit_base', 'vit_large'

    mode : str, optional
        The mode of the model, either 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    -------
    __init__(self, backbone, mode='eval'):
        Initializes the model with the specified backbone and mode.

    predict(self, images):
        Given a batch of images, performs a forward pass through the model and returns predictions.
        Parameters:
        ----------
        images : numpy.ndarray
            A batch of images to perform prediction on, with shape (batch_size, 224, 224, 3).

        Returns:
        -------
        numpy.ndarray
            Model predictions or extracted features for the provided images.
    """

    def __init__(self, backbone, mode='eval', input_shape=(224, 224, 3)):
        self.backbone_name = backbone

        input_layer = Input(shape=input_shape)

        if backbone == 'resnet50':
            self.base_model = ResNet50(weights='imagenet', include_top=False)
        elif backbone == 'resnet101':
            self.base_model = ResNet101(weights='imagenet', include_top=False)
        elif backbone == 'densenet121':
            self.base_model = DenseNet121(weights='imagenet', include_top=False)
        elif backbone == 'densenet169':
            self.base_model = DenseNet169(weights='imagenet', include_top=False)
        elif backbone == 'inception_v3':
            self.base_model = InceptionV3(weights='imagenet', include_top=False)
        elif backbone == 'convnextv2_tiny':
            self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224")
        elif backbone == 'convnextv2_base':
            self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-base-1k-224")
        elif backbone == 'convnextv2_large':
            self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-large-1k-224")
        elif backbone == 'swin_tiny':
            self.base_model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif backbone == 'swin_small':
            self.base_model = TFSwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224")
        elif backbone == 'swin_base':
            self.base_model = TFSwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        elif backbone in ['vit_base', 'vit_large']:
            backbone_path = {
                'vit_base': 'google/vit-base-patch16-224',
                'vit_large': 'google/vit-large-patch16-224',
            }
            self.base_model = TFViTModel.from_pretrained(backbone_path[backbone])
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")


        if mode == 'eval':
            self.base_model.trainable = False

        if backbone in ['vit_base', 'vit_large', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'swin_tiny', 'swin_small', 'swin_base']:
            input_layer_transposed = tf.keras.layers.Permute((3, 1, 2))(input_layer)
            outputs = self.base_model(pixel_values=input_layer_transposed).pooler_output
        else:
            x = self.base_model(input_layer)
            outputs = GlobalAveragePooling2D()(x)

        self.model = Model(inputs=input_layer, outputs=outputs)

    def get_output_shape(self):
        """
        Get the output shape of the model.

        Returns:
        -------
        tuple
            The shape of the model's output tensor.
        """
        return self.model.output_shape

    def predict(self, images):
        """
        Predict on a batch of images.

        Parameters:
        ----------
        images : numpy.ndarray
            A batch of images of shape (batch_size, 224, 224, 3).

        Returns:
        -------
        numpy.ndarray
            Predictions or features from the model for the given images.
        """
        predictions = self.model.predict(images)
        return predictions



class ImageFolderDataset:
    """
    A custom dataset class for loading and preprocessing images from a folder.

    This class helps in loading images from a given folder, automatically filtering valid image files and
    preprocessing them to a specified shape. It also handles any unreadable or corrupted images by excluding them.

    Attributes:
    ----------
    folder_path : str
        The path to the folder containing the images.
    shape : tuple
        The desired shape (width, height) to which the images will be resized.
    image_files : list
        A list of valid image file names that can be processed.

    Parameters:
    ----------
    folder_path : str
        The path to the folder containing image files.
    shape : tuple, optional
        The target shape to resize the images to. The default value is (224, 224).
    image_files : list, optional
        A pre-provided list of image file names. If not provided, it will automatically detect valid image files
        (with extensions '.jpg', '.jpeg', '.png', '.gif') in the specified folder.

    Methods:
    -------
    clean_unidentified_images():
        Cleans the dataset by removing images that cause an `UnidentifiedImageError` during loading. This helps ensure
        that only valid, readable images are kept in the dataset.

    __len__():
        Returns the number of valid images in the dataset after cleaning.

    __getitem__(idx):
        Given an index `idx`, retrieves the image file at that index, loads and preprocesses it, and returns the image
        along with its filename.

    """
    def __init__(self, folder_path, shape=(224, 224), image_files=None):
        self.folder_path = folder_path
        self.shape = shape

        if image_files:
            self.image_files = image_files
        else:
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]

        self.clean_unidentified_images()

    def clean_unidentified_images(self):
        cleaned_files = []
        for img_name in self.image_files:
            img_path = os.path.join(self.folder_path, img_name)
            try:
                Image.open(img_path).convert("RGB")
                cleaned_files.append(img_name)
            except:
                print(f"Skipping {img_name} due to error")

        self.image_files = cleaned_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        img = load_and_preprocess_image(img_path, self.shape)
        return img_name, img

def get_embeddings_df(batch_size=32, path="data/images", dataset_name='', backbone="resnet50", directory='Embeddings', image_files=None):
    """
    Generates embeddings for images in a dataset using a specified backbone model and saves them to a CSV file.
    """

    dataset = ImageFolderDataset(folder_path=path, image_files=image_files)
    model = FoundationalCVModel(backbone)

    img_names = []
    features = []
    num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)

    for i in range(0, len(dataset), batch_size):
        batch_files = dataset.image_files[i:i + batch_size]
        batch_imgs = np.array([dataset[j][1] for j in range(i, min(i + batch_size, len(dataset)))])

        batch_features = model.predict(batch_imgs)

        img_names.extend(batch_files)
        features.extend(batch_features)

        if (i // batch_size + 1) % 10 == 0:
            print(f"Batch {i // batch_size + 1}/{num_batches} done")

    df = pd.DataFrame({
        'ImageName': img_names,
        'Embeddings': features
    })

    df_aux = pd.DataFrame(df['Embeddings'].tolist())
    df = pd.concat([df['ImageName'], df_aux], axis=1)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(f'{directory}/{dataset_name}'):
        os.makedirs(f'{directory}/{dataset_name}')

    df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)

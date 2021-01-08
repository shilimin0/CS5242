import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

def img_paths_labels_from_csv(csv_path, img_dir):
    dir = os.listdir(img_dir)[0]
    img_paths = []
    df = pd.read_csv(os.path.join(img_dir, csv_path))
    for ID in df['ID'].values:
        img_paths.append(str(os.path.join(img_dir,dir, f'{ID}.png')))
    return img_paths, df['Label'].values

def ids_images_from_dir(img_dir):
    IDs = []
    images = []
    for img_file in sorted(os.listdir(img_dir), key=lambda el: int(os.path.splitext(el)[0])):
        IDs.append(os.path.splitext(img_file)[0])
        img = np.array(Image.open(os.path.join(img_dir, img_file)))
        images.append(img)
    return IDs, np.array(images)

# loading images in RGB form
def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    return image, label

"""
Data augmentation:
flip_left_right and flip_up_down help us to increase dataset. It is valid as we could let model to identity the
Lesion area in mutiple manners.
As we obverse from the dataset, some pictures are brighter while some are relatively dark.
random adjust brightness would help to increase the model's ability to recognize the features regardless of brightness
0.3 would be neither too aggressive nor less useless
"""
def augment(image, label, seed=None):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_brightness(image, 0.3, seed=seed)
    return image, label


def create_model():
    tf.keras.backend.clear_session()
    """
    input shape chosen as it is the highest possible before running out of GPU memory.
    """
    base_model = tf.keras.applications.EfficientNetB1(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(480,480,3),
                                                     )
    base_model.trainable = True

    for layer in base_model.layers[:-4]:
        layer.trainable =  False


    """
    Drop-out = 0.5 acceptable for this model from literature research.
    Number of filter = 612, tried 256 but is underfit (augmentation does not improve result / relative high train loss / inability to converge).
    Optimal range seems to be between 512 & 1024. 612 is the lowest to avoid overfitting.
    """
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(612, 1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),#512
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3),
        ]
    )
    return model

def get_class_weight(train_labels):
    a = np.array(train_labels).flatten()
    class_weights = class_weight.compute_class_weight('balanced', np.unique(a),a)
    class_weights = {0:class_weights[0],1:class_weights[1],2:class_weights[2]}
    return class_weights

def creat_train_ds(train_img_paths, train_labels,batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
    train_ds = (train_ds
                .shuffle(len(train_ds))
                .map(parse_function, AUTOTUNE)
                .batch(batch_size)
                .map(augment, AUTOTUNE)
                .prefetch(AUTOTUNE))
    return train_ds

def creat_val_ds(val_img_paths, val_labels,batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_labels))
    val_ds = (val_ds
              .shuffle(len(val_ds))
              .map(parse_function, AUTOTUNE)
              .batch(batch_size)
              .cache()
              .prefetch(AUTOTUNE))
    return val_ds

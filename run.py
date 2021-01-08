import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from code.train import *
import sys

_ , train_path, test_path = sys.argv
CSV_PATH = 'train_label.csv'
TRAIN_IMG_DIR = train_path
TEST_IMG_DIR = test_path
AUTOTUNE = tf.data.experimental.AUTOTUNE
GLOBAL_SEED=42
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
epochs = 50
batch_size = 4
def main():
    train_img_paths, train_labels = img_paths_labels_from_csv(CSV_PATH, TRAIN_IMG_DIR)

    #class distribution is not balanced in the dataset. As we want it not influence the model, so we add on class weight during model training
    class_weights = get_class_weight(train_labels)
    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(train_img_paths,
                                                                                train_labels,
                                                                                test_size=0.04,
                                                                                random_state=GLOBAL_SEED,
                                                                                stratify=train_labels)
    train_ds = creat_train_ds(train_img_paths, train_labels,batch_size)
    val_ds = creat_val_ds(val_img_paths, val_labels,batch_size)
    checkpoint_path = "./ckp/cp-best.ckpt"

    # model check point to save the most not overfitting model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor="val_loss",
                                                     verbose=1,
                                                     period=1)

    # Decays the Learning Rate.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                     patience=2, min_lr=1e-6)

    # Stops if there is no improvement of result after some epochs.
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,
                                                  restore_best_weights=False)
    model = create_model()

    """
    ADAM is used as the default method of optimisation in most papers.
    Not changed throughout the experimentation of the model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=[cp_callback,reduce_lr,stop_early],class_weight =class_weights)
    tf.random.set_seed(GLOBAL_SEED)
    model = create_model()
    model.load_weights('./ckp/cp-best.ckpt')

    IDs, test_images = ids_images_from_dir(os.path.join(TEST_IMG_DIR, os.listdir(TEST_IMG_DIR)[0]))
    test_pred = model.predict(test_images)
    id_col = IDs
    submission_df_1 = pd.DataFrame({"ID": id_col,
                                    "Label": np.argmax(test_pred, axis=1)})
    submission_df_1.head()
    submission_df_1.to_csv('submission_last.csv', index=False)
main()

####### MAIN ######
import argparse
import os

import tensorflow as tf
import numpy      as np
import pandas     as pd

from sklearn.model_selection import train_test_split
from mymodel                 import build_model
from copdgene_data_generator import *

print('Tensorflow version: ' + tf.__version__)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_csv',     required=True, metavar='CSV FILE', help="CSV file pointing to images")
    parser.add_argument('--image_column', required=True, help='Column name for images')
    parser.add_argument('--label_column', required=True, help='Column name for labels')
    parser.add_argument('--test_ratio',   help='Percentage for testing data. Default is 0.3 (30%)', type=float,
                        default=0.3)
    parser.add_argument('--epochs',     help='Number of epochs. Default is 15',   type=int,      default=15)
    parser.add_argument('--batch_size', help='Training batch size. Default is 8', type=int,      default=8)
    parser.add_argument('--output',     help="Specify file name for output. Default is 'model'", default='model')
    args = parser.parse_args()

    epochs     = args.epochs
    batch_size = args.batch_size
    output     = args.output
    test_ratio = args.test_ratio

    # Point to images
    image_list_file = args.data_csv
    image_column    = args.image_column
    label_column    = args.label_column

    # Pull the list of files
    train_df = pd.read_csv(image_list_file)
    images   = train_df[image_column].to_list()
    labels   = train_df[label_column].to_list()

    # Split test set
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_ratio,
                                                                            random_state=42)

    # FOR DEBUG REMOVE IT
    print(f"Train shape: {len(train_images)}")
    print(f"Train label length: {len(train_labels)}")
    print(train_images[:2])
    print(train_labels[:2])

    print(f"Test shape: {len(test_images)}")
    print(f"Test label length: {len(test_labels)}")

    # Get total number of images in each set
    train_image_sizes, train_image_count = getImageSetSize(train_images)
    test_image_sizes,  test_image_count  = getImageSetSize(test_images)

    # FOR DEBUG REMOVE IT
    print(f"train_image_sizes: {train_image_sizes}")
    print(f"train_image_count: {train_image_count}")

    print(f"test_image_sizes: {test_image_sizes}")
    print(f"test_image_count: {test_image_count}")

    # Create a mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # Initialize settings for training
    train_steps = train_image_count // batch_size
    val_steps   = test_image_count // batch_size

    # FOR DEBUG REMOVE IT
    print(f"train_steps: {train_steps}")
    print(f"val_steps: {val_steps}")

    # # Create the data generators
    trainGen = batchGenerator(train_images, train_labels, batch_size)
    testGen  = batchGenerator(test_images,  test_labels,  batch_size)

    # # Build the model
    classes     = 5
    loss_type   = 'sparse_categorical_crossentropy'
    lst_metrics = ['sparse_categorical_accuracy']
    lr_rate     = 1e-9

    with strategy.scope():
        model = build_model(image_dims=(512,512,1), n_slices=20)
        opt   = tf.keras.optimizers.Adam(learning_rate=lr_rate)
        model.compile(loss=loss_type, optimizer=opt, metrics=lst_metrics)

    # Print Model Summary
    print(model.summary())

    # Train the model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(output + '.h5', verbose=1,
                                                          monitor='val_sparse_categorical_accuracy',
                                                          save_best_only=True, mode='max')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=7,
                                                     verbose=0, mode='max')
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, mode='min')

    H = model.fit(x=trainGen,
                  steps_per_epoch=train_steps,
                  validation_data=testGen,
                  validation_steps=val_steps,
                  epochs=epochs,
                  callbacks=[model_checkpoint, earlyStopping, reduceLR])

    # Save loss history
    loss_history = np.array(H.history['loss'])
    np.savetxt(output + '_loss.csv', loss_history, delimiter=",")
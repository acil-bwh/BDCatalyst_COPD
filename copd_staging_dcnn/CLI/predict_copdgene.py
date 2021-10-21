####### MAIN ######
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd

from mymodel import build_model
from copdgene_data_generator import *
from sklearn.metrics         import confusion_matrix
import os

from utils import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from tf.keras.callbacks import ModelCheckpoint
print('Tensorflow version: ' + tf.__version__)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', metavar='FOLDER', required=True, help="Directory containing images.")
    parser.add_argument('--insp_exp', help='Specify INSP(iration) or EXP(iration). Default is both', default='')
    parser.add_argument('--std_sharp', help='Specify STD or SHARP images. Default is both',          default='')
    parser.add_argument('--num_files', help='Number of files to include in testing. Default is 100', type=int,
                        default=100)
    parser.add_argument('--trained_model', help='Trained Model .h5 file',            required=True)
    parser.add_argument('--batch_size',    help='Training batch size. Default is 8', type=int, default=8)
    parser.add_argument('--num_classes',   help='Number of classes',                 type=int, default=1)
    parser.add_argument('--name_analysis', help="Name your analysis test outputs",   default="Test-Analysis_")

    args = parser.parse_args()

    insp_exp      = args.insp_exp
    std_sharp     = args.std_sharp
    num_files     = args.num_files
    batch_size    = args.batch_size
    trained_model = args.trained_model
    num_classes   = args.num_classes
    name_analysis = args.name_analysis

    class_threshold = 0.5

    parameters_dict = {"name_analysis": name_analysis}

    # Point to project files folder
    parent_dir = args.data_directory

    # Pull the list of files
    test_images, test_labels = pullRandomNrrds(parent_dir, num_files=num_files)

    # FOR DEBUG REMOVE IT
    eprint(f"Test shape: {len(test_images)}")
    eprint(f"Test label len: {len(test_labels)}")

    # Get total number of images in each set
    test_image_sizes, test_image_count = getImageSetSize(test_images)

    # FOR DEBUG REMOVE IT
    eprint(f"test_image_sizes: {test_image_sizes}")
    eprint(f"test_image_count: {test_image_count}")

    # Create a mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    eprint(f'Number of devices: {strategy.num_replicas_in_sync}')

    # # Create the data generators
    testGen = batchGeneratorPredict(test_images, test_labels, batch_size=batch_size, small_set=test_image_count)
    #   img_array, img_label, file_index = processNextNrrd(train_images, train_labels, 0)

    with strategy.scope():
        eprint('Loading model...')
        model = tf.keras.models.load_model(trained_model)

        eprint('Calculating metrics...')
        preds = model.predict_generator(testGen)

    # probs_preds = np.argmax(preds, axis=1)
    results = [1 if r > class_threshold else 0 for i, r in enumerate(preds)]

    eprint('[DEBUG] Print Prediction Probabilies')
    eprint(results)

    test_input = [0] * (test_image_count - 32) + [1] * 32

    eprint('[PROGRESS] Calculating metrics...')

    eprint('[1] Calculating plot_auc...')
    plot_auc(test_input, preds, parameters_dict)

    eprint('[2] Calculating plot_predictions...')
    plot_predictions(test_input, results, parameters_dict)

    eprint('[3] Calculating calculate_metrics...')
    d = {}
    d['Test accuracy'], d['Test AUC'], d['Test AUPRC'], d['Test precision'], d['Test recall'], d['Test F-score'], d[
        'Test MCC'] = calculate_metrics(test_input, preds, results)

    eprint('[4] Calculating plot_confusion_matrix...')
    cm_val = confusion_matrix(test_input, results)
    d['Test specificity'], d['Test fall-out'] = plot_confusion_matrix(cm_val, [0, 1], parameters_dict,
                                                                      "Confusion Matrix")

    eprint('[5] Calculating plot_confusion_matrix...')
    conf_col = ':'.join(parameters_dict.keys())
    columns  = [conf_col] + list(d.keys())
    df_out   = pd.DataFrame(columns=columns)
    df_out.loc[0] = [':'.join(map(str, parameters_dict.values()))] + list(d.values())
    df_out.to_csv(name_analysis + '.csv', index=False)
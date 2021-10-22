####### MAIN ######
import argparse
import os

import tensorflow as tf
import numpy      as np
import pandas     as pd
import seaborn    as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix, balanced_accuracy_score, classification_report
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
    
    
    # CALCULATING METRICS ON TESTING SET:
    print('Calculating metrics...')

    # Load best model:
    model = tf.keras.models.load_model(output + '.h5')
    # Make predictions using the testing generator:
    predictions = model.predict(testGen, batch_size=batch_size, verbose=1)

    # Calculate the overall accuracy and balanced accuracy of the model:
    acc   = accuracy_score(test_labels, np.argmax(predictions, axis=-1))
    b_acc = balanced_accuracy_score(test_labels, np.argmax(predictions, axis=-1))
    print('Accuracy:          %.2f%%' % (100 * acc))
    print('Balanced accuracy: %.2f%%' % (100 * b_acc))

    # Get the classification report of the model:
    index  = ['GOLD 0', 'GOLD 1', 'GOLD 2', 'GOLD 3', 'GOLD 4']
    report = classification_report(test_labels, np.argmax(predictions, axis=-1), target_names=index, zero_division=0)
    print(report)

    # Calculate the normalized confusion matrix:
    c_m   = confusion_matrix(test_labels, np.argmax(predictions, axis=-1))

    # Normalize the confusion matrix by true label:
    c_m[0, :] = c_m[0, :] / len(test_labels[test_labels == 0]) * 100
    c_m[1, :] = c_m[1, :] / len(test_labels[test_labels == 1]) * 100
    c_m[2, :] = c_m[2, :] / len(test_labels[test_labels == 2]) * 100
    c_m[3, :] = c_m[3, :] / len(test_labels[test_labels == 3]) * 100
    c_m[4, :] = c_m[4, :] / len(test_labels[test_labels == 4]) * 100
    cm_df = pd.DataFrame(c_m,
                         index=index,
                         columns=index)

    plt.figure(figsize=(8, 5), dpi=120)
    sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='.2f', cmap='Spectral_r',
                square=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Balanced accuracy = {:.3f}'.format(b_acc))
    plt.yticks(rotation=0)
    plt.savefig(output + '_ConfusionMatrix.png', transparent=True, bbox_inches='tight', pad_inches = 0) 
    plt.show()

    # Calculate and plot the AUC and the ROC curves:
    lr_probs = np.copy(predictions)
    lr_auc   = roc_auc_score(test_labels[:], lr_probs, multi_class='ovr')
    print('DCNN: AUC ROC = %.3f' % (lr_auc))

    # Compute the AUC ROC for each of the GOLD scores:
    lr_auc = roc_auc_score(np.eye(5)[test_labels.astype(int)][:, 0], lr_probs[:, 0])
    print('- Gold 0: %.3f' % (lr_auc))
    lr_auc = roc_auc_score(np.eye(5)[test_labels.astype(int)][:, 1], lr_probs[:, 1])
    print('- Gold 1: %.3f' % (lr_auc))
    lr_auc = roc_auc_score(np.eye(5)[test_labels.astype(int)][:, 2], lr_probs[:, 2])
    print('- Gold 2: %.3f' % (lr_auc))
    lr_auc = roc_auc_score(np.eye(5)[test_labels.astype(int)][:, 3], lr_probs[:, 3])
    print('- Gold 3: %.3f' % (lr_auc))
    lr_auc = roc_auc_score(np.eye(5)[test_labels.astype(int)][:, 4], lr_probs[:, 4])
    print('- Gold 4: %.3f' % (lr_auc))

    # Calculate and plot the ROC curves:
    ns_fpr, ns_tpr, _     = roc_curve(np.eye(5)[test_labels.astype(int)][:, 0], np.zeros(len(test_labels)))
    lr_fpr_0, lr_tpr_0, _ = roc_curve(np.eye(5)[test_labels.astype(int)][:, 0], lr_probs[:, 0])
    lr_fpr_1, lr_tpr_1, _ = roc_curve(np.eye(5)[test_labels.astype(int)][:, 1], lr_probs[:, 1])
    lr_fpr_2, lr_tpr_2, _ = roc_curve(np.eye(5)[test_labels.astype(int)][:, 2], lr_probs[:, 2])
    lr_fpr_3, lr_tpr_3, _ = roc_curve(np.eye(5)[test_labels.astype(int)][:, 3], lr_probs[:, 3])
    lr_fpr_4, lr_tpr_4, _ = roc_curve(np.eye(5)[test_labels.astype(int)][:, 4], lr_probs[:, 4])

    plt.figure(figsize=(8, 8))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr_0, lr_tpr_0, marker=',', label='Gold_0')
    plt.plot(lr_fpr_1, lr_tpr_1, marker=',', label='Gold_1')
    plt.plot(lr_fpr_2, lr_tpr_2, marker=',', label='Gold_2')
    plt.plot(lr_fpr_3, lr_tpr_3, marker=',', label='Gold_3')
    plt.plot(lr_fpr_4, lr_tpr_4, marker=',', label='Gold_4')
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves', fontweight='bold')
    plt.legend()
    plt.savefig(output + '_ROCcurves.png', transparent=True, bbox_inches='tight', pad_inches = 0) 
    plt.show()

    # Plot the probability distribution of each class:
    labels = []
    for ii in np.arange(0, len(lr_probs)):
        if test_labels[ii] == 0:
            labels.append('Gold_0')
        elif test_labels[ii] == 1:
            labels.append('Gold_1')
        elif test_labels[ii] == 2:
            labels.append('Gold_2')
        elif test_labels[ii] == 3:
            labels.append('Gold_3')
        elif test_labels[ii] == 4:
            labels.append('Gold_4')

    idx_max       = np.expand_dims(np.argmax(lr_probs,    axis=1), -1)
    curated_preds = np.take_along_axis(lr_probs, idx_max, axis=1).squeeze()
    df = pd.DataFrame({'Predictions': 100 * curated_preds,
                       'Labels': labels})

    # Plot the probability distribution of each class:
    plt.figure(figsize=(8, 6))
    plt.grid()
    sns.boxplot(x='Labels', y='Predictions', data=df, order=['Gold_0', 'Gold_1', 'Gold_2', 'Gold_3', 'Gold_4'])
    plt.savefig(output + '_ProbabilityDistribution.png', transparent=True, bbox_inches='tight', pad_inches = 0) 
    plt.show()

    ll  = df['Labels'] == 'Gold_0'
    p_0 = df['Predictions'][ll == True]
    ll  = df['Labels'] == 'Gold_1'
    p_1 = df['Predictions'][ll == True]
    ll  = df['Labels'] == 'Gold_2'
    p_2 = df['Predictions'][ll == True]
    ll  = df['Labels'] == 'Gold_3'
    p_3 = df['Predictions'][ll == True]
    ll  = df['Labels'] == 'Gold_4'
    p_4 = df['Predictions'][ll == True]

    print('GOLD 0:  perc25 = %.3f   perc75 = %.3f' % (np.percentile(p_0, 25), np.percentile(p_0, 75)))
    print('GOLD 1:  perc25 = %.3f   perc75 = %.3f' % (np.percentile(p_1, 25), np.percentile(p_1, 75)))
    print('GOLD 2:  perc25 = %.3f   perc75 = %.3f' % (np.percentile(p_2, 25), np.percentile(p_2, 75)))
    print('GOLD 3:  perc25 = %.3f   perc75 = %.3f' % (np.percentile(p_3, 25), np.percentile(p_3, 75)))
    print('GOLD 4:  perc25 = %.3f   perc75 = %.3f' % (np.percentile(p_4, 25), np.percentile(p_4, 75)))

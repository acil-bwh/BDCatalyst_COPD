from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, \
                            precision_recall_fscore_support, matthews_corrcoef
import sys
import numpy  as np
import itertools
import matplotlib.pyplot as plt


def eprint(args):
    sys.stderr.write(str(args) + "\n")

# Print History in detail
# TRAINING
def print_history(history, validation):
    epochs = len(history.history['loss'])
    for epoch in range(epochs):
        if validation:
            eprint('Epoch {}/{}: accuracy : {:.3f}, loss : {:.3f}, val_accuracy : {:.3f}, val_loss : {:.3f}'.format(
                epoch + 1, epochs, history.history['accuracy'][epoch], history.history['loss'][epoch],
                history.history['val_accuracy'][epoch], history.history['val_loss'][epoch]))
        else:
            eprint('Epoch {}/{}: accuracy : {:.3f}, loss : {:.3f} '.format(
                epoch + 1, epochs, history.history['accuracy'][epoch], history.history['loss'][epoch]))

# TRAINING OR TESTING
def plot_auc(y_labels, pred, parameters_dict = {}, title=''):
    # Scores
    false_positive_rate, recall, thresholds = roc_curve(y_labels, pred)
    roc_auc = auc(false_positive_rate, recall)

    # Plot
    plt.figure()
    plt.title(title + ' Receiver Operating Characteristic (ROC)')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out (1-Specificity)')

    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig(title + '_AUC_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def calculate_metrics(y_labels, prob, pred, average="binary", num_classes = 1):
    if num_classes == 2:
        roc_auc = roc_auc_score(y_labels, prob[:, 1], multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_labels, prob, multi_class='ovr')

    # loss = log_loss(y_labels, prob, eps=1e-15, normalize=True)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_labels, pred, average=average)

    precision_array, recall_array, _, _ = precision_recall_fscore_support(y_labels, pred, average=None)
    sorted_pairs = sorted(zip(recall_array, precision_array), key=lambda x: x[0])
    tuples = zip(*sorted_pairs)
    recall_array, precision_array = [list(tuple) for tuple in tuples]
    auprc = auc(recall_array, precision_array)

    acc = accuracy_score(y_labels, pred)
    mcc = matthews_corrcoef(y_labels, pred)

    return np.round(100 * acc, 3), \
           np.round(100 * roc_auc, 3), \
           np.round(100 * auprc, 3), \
           np.round(100 * precision, 3), \
           np.round(100 * recall, 3), \
           np.round(100 * fscore, 3), \
           np.round(mcc, 3)

# TRAINING
def plot_loss(history, parameters_dict = {}):
    loss_train     = list(np.log10(history.history['loss']))
    loss_val       = list(np.log10(history.history['val_loss']))
    epochs_initial = len(loss_val) #_initial)

    epochs   = range(1, epochs_initial + 1)
    min_loss = min(loss_train + loss_val)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training vs Validation loss')
    plt.axvline(x=epochs_initial, c='red', ymax=0.99, ymin=0.01, linestyle='--')
    plt.xticks(epochs)
    plt.text(x=epochs_initial - 0.8, y=min_loss - 0.15, s='initial', c='red')
    plt.xlabel('Epochs')
    plt.ylabel('log(Loss)')
    plt.legend(loc='best')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':','_').replace(',', '-')
    plt.savefig('Loss_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def plot_predictions(validation_labels, pred, parameters_dict = {}, title=''):
    num_samples_to_show = np.min([len(pred), 100])
    plt.figure(figsize=(30, 10))
    plt.plot(range(num_samples_to_show), pred[:num_samples_to_show], 'ys', label='Predicted_value')
    plt.plot(range(num_samples_to_show), validation_labels[:num_samples_to_show], 'r*', label='Test_value')

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('predicted')
    plt.xlabel(' samples')
    plt.legend(loc="best")
    plt.title(title + ' - Truth vs predicted')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig(title + '_Predictions_' + conf + '.png')
    plt.show()

# TRAINING OR TESTING
def plot_confusion_matrix(cm, class_names, parameters_dict = {}, title=''):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title.replace('_', ' '))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix
    labels = np.around(cm.astype('int'))  # / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    conf = ':'.join(map(str, parameters_dict.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')
    plt.savefig(title + conf + '.png')

    # Calculate specificity and fall-out for each class
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    TN = TN.astype(float)

    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # Fall out or false positive rate
    fallout = 1 - specificity

    # Return mean value - to discuss
    return np.round(100 * specificity.mean(), 3), \
           np.round(100 * fallout.mean(), 3)
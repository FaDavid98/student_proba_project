from matplotlib import pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy import interp
from itertools import cycle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

def plotting_loss(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plotting_acc(history):
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def conf_matrix(YTest, y_pred):
    YTest=to_categorical(YTest)
    plt.figure(3)
    mat = confusion_matrix(YTest.argmax(axis=1), y_pred.argmax(axis=1))
    print(mat)
    matrix = classification_report(YTest.argmax(axis=1), y_pred.argmax(axis=1))
    print('Classification report : \n',matrix)
    sns.heatmap(mat, annot=True)
    nn=plt.title("Neural network conf matrix")
    plt.show()
    return nn

def conf_matrix2(YTest, y_pred):
    plt.figure(3)
    mat = confusion_matrix(YTest.argmax(axis = 1), y_pred.argmax(axis = 1))
    print(mat)
    matrix = classification_report(YTest.argmax(axis=1), y_pred.argmax(axis=1))
    print('Classification report : \n',matrix)
    sns.heatmap(mat, annot=True)
    oo=plt.title("Conf matrix")
    plt.show()
    return oo

def ROC_curve(YTraining, YTest, y_score):                   

    # n_classes = YTraining.shape[1]
    # Compute ROC curve and ROC area for each class
    YTest=to_categorical(YTest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(YTest[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])


    # Plot all ROC curves
    plt.figure()
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Neural network ROC curves')
    plt.legend(loc="lower right")
    plt.show()


def ROC_curve2(YTraining, YTest, y_score):                   

    # n_classes = YTraining.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(YTest[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])


    # Plot all ROC curves
    plt.figure()
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()

    
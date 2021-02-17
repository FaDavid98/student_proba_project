from config import *
import Preprocesing
import feature_selection
import Classification
import drawing

file_name = FILE_NAME
number_of_features = NUMBER_OF_FEATURES
test_size = TEST_SIZE
epochs = EPOCHS
learning_rate = LEARNING_RATE

#read file
df = Preprocesing.load(file_name)

#converting features
df = Preprocesing.convert(df)

#descending categories
df = Preprocesing.descending_class(df)

#selecting x,y from table
x,y = Preprocesing.selection(df)

#trin test split
XTraining, XTest, YTraining, YTest=Preprocesing.split(x,y,test_size)

#normalization
XTraining, XTest = Preprocesing.normalization(x,XTraining, XTest)

#feature selection
XTraining, XTest, m = feature_selection.chi_squared(XTraining, YTraining, XTest,number_of_features)

#classification
'''This is for classificator: neural network'''
loss, acc, history, y_pred, y_score = Classification.neural_network(XTraining, YTraining, XTest, YTest, epochs, m, learning_rate)
#plotting
drawing.plotting_loss(history)
drawing.plotting_acc(history)
nn=drawing.conf_matrix(YTest, y_pred)
drawing.ROC_curve(YTraining, YTest, y_score)

'''These are for other classificators'''
y_pred, y_score ,YTest = Classification.randomforest(XTraining, YTraining, XTest, YTest)
#plotting ROC curve
drawing.ROC_curve2(YTraining, YTest, y_score)
#confusion matrix
oo=drawing.conf_matrix2(YTest, y_pred)


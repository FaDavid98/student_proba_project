import tensorflow as tf
from sklearn import metrics
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
np.random.seed(8)
tf.random.set_seed(10)

def neural_network(XTraining, YTraining, XTest, YTest, epochs, m, learning_rate):
    #categorization
    YTest=to_categorical(YTest)
    YTraining=to_categorical(YTraining)
    #model
    model = tf.keras.models.Sequential()
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.add(tf.keras.layers.Dense(10, input_shape=(m,), activation='relu'))
    model.add(tf.keras.layers.Dense(3 ,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(XTraining, YTraining, epochs=epochs, validation_split=0.2)
    y_pred = np.round(model.predict(XTest,2))
    y_score = model.predict_proba(XTest)
    lossfunc, acc = model.evaluate(XTest, YTest, verbose=2)
    print('Neural network accuracy: %.2f' % (acc*100))
    return lossfunc, acc, history, y_pred, y_score

def decisiontree(XTraining, YTraining, XTest, YTest):
    clf=DecisionTreeClassifier()
    clf.fit(XTraining,YTraining)
    y_pred=clf.predict(XTest)
    y_score=clf.predict_proba(XTest)
    print('Decision tree accuracy:', metrics.accuracy_score(YTest,y_pred)*100)
    YTest=label_binarize(YTest, classes=[0,1,2,3])
    y_pred = label_binarize(y_pred, classes=[0,1,2,3])
    return y_pred, y_score, YTest

def randomforest(XTraining, YTraining, XTest, YTest):
    clf=RandomForestClassifier(n_estimators=1000)
    clf.fit(XTraining,YTraining)
    y_pred=clf.predict(XTest)
    y_score=clf.predict_proba(XTest)   
    print('Random forest accuracy:',metrics.accuracy_score(YTest, y_pred)*100)
    YTest=label_binarize(YTest, classes=[0,1,2,3])
    y_pred = label_binarize(y_pred, classes=[0,1,2,3])
    return y_pred, y_score, YTest

def logisticregression(XTraining, YTraining, XTest, YTest):
    classifier = LogisticRegression(max_iter=800)
    clf = classifier.fit(XTraining, YTraining)
    y_pred = clf.predict(XTest)
    y_score=clf.predict_proba(XTest)
    print('Logistic regression accuracy:',metrics.accuracy_score(YTest, y_pred)*100)
    YTest=label_binarize(YTest, classes=[0,1,2,3])
    y_pred = label_binarize(y_pred, classes=[0,1,2,3])
    return y_pred, y_score, YTest



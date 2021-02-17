from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2

def new_features(features, XTest):
    names = XTest.columns
    usable = features.get_support() #list of booleans
    new_features = [] # The list of your K best features
    for bool, feature in zip(usable, names):
        if bool:
            new_features.append(feature)
    XTest = XTest[new_features]
    m = XTest.shape[1] #geting input for neural network
    return XTest, m

def chi_squared(XTraining, YTraining, XTest, number_of_features):
    chi2_features = SelectKBest(chi2, k=number_of_features) 
    select_k_best_classifier = chi2_features.fit(XTraining, YTraining) 
    XTraining = chi2_features.fit_transform(XTraining, YTraining) 
    XTest, m = new_features(select_k_best_classifier, XTest)
    return XTraining, XTest, m

def recursive_feature_elimination(XTraining, YTraining, XTest, number_of_features):
    rfe_selector=RFE(LogisticRegression(max_iter=10000), n_features_to_select=number_of_features)
    recursive_selected = rfe_selector.fit(XTraining, YTraining) 
    XTraining=rfe_selector.fit_transform(XTraining,YTraining)
    XTest, m = new_features(recursive_selected, XTest)
    return XTraining, XTest, m

def lasso(XTraining, YTraining, XTest, number_of_features):
    embeded_lr_selector = SelectFromModel(LogisticRegression(max_iter= 1000),max_features=number_of_features)
    lasso_selected = embeded_lr_selector.fit(XTraining,YTraining)
    XTraining = embeded_lr_selector.fit_transform(XTraining ,YTraining)
    XTest, m = new_features(lasso_selected, XTest)
    return XTraining, XTest, m

def randomforest(XTraining, YTraining, XTest, number_of_features):
    embeded_lr_selector = SelectFromModel(RandomForestClassifier(n_estimators=100),max_features=number_of_features)
    randomforest_selected = embeded_lr_selector.fit(XTraining,YTraining)
    XTraining = embeded_lr_selector.fit_transform(XTraining,YTraining) 
    XTest, m = new_features(randomforest_selected, XTest)
    return XTraining, XTest, m 

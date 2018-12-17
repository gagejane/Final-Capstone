import code_for_analysis as an
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':

    '''Run Decision Tree Models with count features and binary features to find the better model'''
    #
    print('*****MODELING WITH COUNT FEATURES*****')
    df_count = pd.read_csv('data/df_merged_count.csv', low_memory=False)

    df_count.drop(['gtd_gname', 'year'], axis=1, inplace=True)
    y = df_count.pop('suicide').values
    X = df_count.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)
    print(y_test.shape)
    print(y_train.shape)
    print(y.shape)

    an.df_models(X_train, y_train, X_test, y_test)

    print('*****MODELING WITH BINARY FEATURES*****')
    df_binary = pd.read_csv('data/df_merged_binary.csv', low_memory=False)

    df_binary.drop(['gtd_gname', 'year'], axis=1, inplace=True)
    y = df_binary.pop('suicide').values
    X = df_binary.values
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y, stratify = y)
    print(y_test_bin.shape)

    an.df_models(X_train_bin, y_train_bin, X_test_bin, y_test_bin)



    '''Model with count features is better; run model through other algorithms'''

    '''Run lines 35-63 to compare algorithms and create lists of items for ROC plot'''

    '''Lists used to generate ROC curve plot'''
    algo_name_list = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'AdaBoosting']
    y_predict_list = []
    algo_for_predict_proba_list = []
    #
    '''Decision Tree'''
    y_predict, dt = an.d_tree(X_train, y_train, X_test, y_test)
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(dt)
    #
    '''Logistic Regression'''
    y_predict, lgt = an.logit(X_train, y_train, X_test, y_test)
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(lgt)
    #
    '''Random Forests'''
    rf_cnf_matrix, accuracy, y_predict, importances, rfc = an.rf(X_train, y_train, X_test, y_test)
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(rfc)
    #
    '''Gradient Boosting'''
    y_predict, gdbc = an.gdb(X_train, y_train, X_test, y_test)
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(gdbc)

    '''AdaBoosting'''
    y_predict, ada = an.ada(X_train, y_train, X_test, y_test)
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(ada)
    #
    '''Generate ROC plot'''
    an.roc_plot(algo_name_list, y_predict_list, algo_for_predict_proba_list, y_test, X_test)
    #

    '''Return obects for plotting confusion matrix and feature importances, derived from RF model'''
    rf_cnf_matrix, accuracy, y_predict, importances, rfc = an.rf(X_train, y_train, X_test, y_test)
    class_names = ['Not suicide', 'Suicide']

    '''Plot normalized and not normalized confusion matrices'''
    an.plot_confusion_matrix(rf_cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix using Random Forests')
    an.plot_confusion_matrix(rf_cnf_matrix, classes=class_names, title='Confusion Matrix, Without Normalization, Using Random Forests')
    #
    '''Plot feature importances'''
    an.feat_imp(importances)

    '''Run t tests to pintpoint differences'''
    an.ttests()

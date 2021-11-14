from sklearn.metrics import classification_report, precision_recall_fscore_support

def run_model_training(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train,y_train)
    classifier.score(X_test,y_test)
    y_pred = classifier.predict(X_test)
    
    # Get precision and recall 
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                      average = 'weighted', 
                                                      beta=0.5)

    return classifier, prec, rec




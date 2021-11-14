# Packages
import mlflow 
import pandas as pd
# Get customized functions from library
import packages.data_processor as dp
import packages.model_trainer as mt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 0.Path to data
path_to_data = './data/spam.csv'

# 1.Prepare the data
prepared_data = dp.prepare_data(path_to_data, encoding="latin-1")

# 2.Create train - test split
train_test_data = dp.create_train_test_data(prepared_data['text'], 
                                         prepared_data['label'], 
                                         0.33, 2021)

# 3. Candiate Models 
random_forest = RandomForestClassifier(class_weight ='balanced', criterion = 'gini', max_depth = 11, 
                                   n_estimators= 10, random_state= 0)

logistic_regression = LogisticRegression(penalty='l1', solver='saga', fit_intercept=True)

# 4. Train & Experiment
if __name__ == "__main__":

    mlflow.set_experiment(experiment_name = "MlFlow: Spam Classifiers")

    # Info about the first model
    classifier, prec, rec = mt.run_model_training(random_forest, 
                                                  train_test_data['x_train'], 
                                                  train_test_data['x_test'], 
                                                  train_test_data['y_train'], 
                                                  train_test_data['y_test'])
    mlflow.sklearn.log_model(classifier, "Random Forest")
    mlflow.log_metric("Random Forest Precision", prec)
    mlflow.log_metric("Random Forest Recall", rec)

     # Info about the second model
    classifier, prec, rec = mt.run_model_training(logistic_regression, 
                                                  train_test_data['x_train'], 
                                                  train_test_data['x_test'], 
                                                  train_test_data['y_train'], 
                                                  train_test_data['y_test'])
    
    mlflow.sklearn.log_model(classifier, "Logistic Regression")
    mlflow.log_metric("Logistic Regression Precision", prec)
    mlflow.log_metric("Logistic RegressionRecall", rec)
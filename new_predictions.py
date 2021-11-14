import mlflow
import pandas as pd
logged_model = 'runs:/08174ea1631a4a6c96df6f7d69f96f9c/Random Forest'

# 0.Path to data
path_to_data = './data/spam.csv'

samples = pd.read_csv(path_to_data).sample
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))
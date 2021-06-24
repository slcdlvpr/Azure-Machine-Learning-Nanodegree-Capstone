
import pandas as pd
import numpy as np
import argparse
import os
import joblib
from azureml.core.run import Run
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#azure libraries 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import TabularDatasetFactory


#skleanr libraries 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import joblib

# get the path to the data 
#data_path = "https://ml.azure.com/fileexplorerAzNB?wsid=/subscriptions/61c5c3f0-6dc7-4ed9-a7f3-c704b20e3b30/resourcegroups/aml-quickstarts-147467/workspaces/quick-starts-ws-147467&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&activeFilePath=Users/odl_user_147467/train.csv"

#subscription_id = '61c5c3f0-6dc7-4ed9-a7f3-c704b20e3b30'
#resource_group = 'aml-quickstarts-147467'
#workspace_name = 'quick-starts-ws-147467'
#workspace = Workspace(subscription_id, resource_group, workspace_name)
#dataset = Dataset.get_by_name(workspace, name='crossell')
#ds = dataset.to_pandas_dataframe()
             
#data_path = 'https://mlstrg147518.file.core.windows.net/code-391ff5ac-6576-460f-ba4d-7e03433c68b6/Users/odl_user_147518/train.csv'
#ds = TabularDatasetFactory.from_delimited_files(data_path)


#subscription_id = '3e42d11f-d64d-4173-af9b-12ecaa1030b3'
#resource_group = 'aml-quickstarts-147623'
#workspace_name = 'quick-starts-ws-147623'

#get the run context   
run = Run.get_context()
workspace = run.experiment.workspace
dataset = Dataset.get_by_name(workspace, name='CrossSell Dataset')


#dataset.to_pandas_dataframe()  alternative method 
  
# load clean data and get the target column data 
def clean_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    x_df["Gender"] = x_df.Gender.apply(lambda s: 1 if s == "Male" else 0)
    x_df["Vehicle_Damage"] = x_df.Vehicle_Damage.apply(lambda s: 1 if s == "Yes" else 0)
    x_df["Vehicle_Age"] = x_df.Vehicle_Age.apply(lambda s: 2 if s == "> 2 Years" else 1)
    y_df = x_df.pop("Response").apply(lambda s: 1 if s == 1 else 0)

    return x_df, y_df

# Splitting data into train and test sets

#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test) 

def main():
# Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    x, y = clean_data(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()




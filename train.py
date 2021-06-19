import pandas as pd
import numpy as np
import argparse
import os
import joblib
from azureml.core.run import Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import StandardScaler

# get the path to the data 
#data_path = "https://ml.azure.com/fileexplorerAzNB?wsid=/subscriptions/61c5c3f0-6dc7-4ed9-a7f3-c704b20e3b30/resourcegroups/aml-quickstarts-147467/workspaces/quick-starts-ws-147467&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&activeFilePath=Users/odl_user_147467/train.csv"

#subscription_id = '61c5c3f0-6dc7-4ed9-a7f3-c704b20e3b30'
#resource_group = 'aml-quickstarts-147467'
#workspace_name = 'quick-starts-ws-147467'
#workspace = Workspace(subscription_id, resource_group, workspace_name)
#dataset = Dataset.get_by_name(workspace, name='crossell')
#ds = dataset.to_pandas_dataframe()

data_path = 'train.csv'
ds = TabularDatasetFactory.from_delimited_files(data_path)

#get the run context 
run = Run.get_context()

# load clean data and get the target column data 
x_df = ds.to_pandas_dataframe().dropna()
y_df = x_df.pop("Response")


# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

def main():
    # setup the parsed script arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=20, help="The number of trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node")
    
    #get passed args 
    args = parser.parse_args(args=[])
    run.log("The number of trees in the forest:", np.int(args.n_estimators))
    run.log("The minimum number of samples required to split an internal node:", np.int(args.min_samples_split))
    
    #setup
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, min_samples_split=args.min_samples_split).fit(x_train, y_train)

    #process and score 
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

    # Saving model to the output folder 
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()




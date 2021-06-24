*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

The project uses Insurance Crosssell  Dataset in azure workspace to train models using both Hyperdrive and AutoML.  The best model is determined and then a endpoint is created and tested using python sdk. The project was implemented using VS code and Azure Machine Learning Studio. 

## Project Set Up and Installation
The project requires access to a Azure Machine Learning account and the crosssell dataset.  Link to the dataset is provided later in this document.

## Dataset

### Overview
Each row in the dataset represents a policy holder with the insurance company that is a potential sale of the companies car insurance product.  
The data includes:
    * Customer demographics 
    * Age and condition of the customers vehicle 
    * Is the customer a current policy holder 

### Task
Ths client is an Insurance company that has provided Health Insurance to its customers now they want a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.


### Access
Data is accessed via a dataset after it is downloaded from Kaggle and uploaded into the Azure ML studio. 
Raw data can be accessed here <a href = "https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction">Crosssell Dataset</a>
## Automated ML
In order to setup the automl run the following tasks were performed:

* Specify the primary_metric: The metric that will be used is Accuracy.  We want to select the model that will most accurately predict if the customer will be interested in the car insurance
* Set experiment_timeout_minutes (20): In order to limit consumed resources we set a max time out for the experiment. 
* Enable_early_termination (True):  Abandon models that are not more accurate than currently completed models
* Max_concurrent_iterations (5): Represents the maximum number of iterations that would be executed in parallel

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

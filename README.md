
# Insurance Crosssell Capstone

The project uses Insurance Crosssell  Dataset in azure workspace to train models using both Hyperdrive and AutoML.  The best model is determined and then a endpoint is created and tested using python sdk. The project was implemented using VS code and Azure Machine Learning Studio. 

# Project Architecture 
Below is a high level overview of the project architecture.  

![Architecture](Images/Automl/TopImage.JPG)

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
 Below is a screeen shot of the run widget as the run was performed. 
 
![Widget](Images/Automl/Widget.JPG)

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
Below shows the results of our run.  As you can see our primary metric (Accuracy) varies by the model tested. Also included is a screen shot of the best model complete
with parameters. The bottom screen shot shows hyper parameters used with the model. 

![Models](Images/Automl/DifferentModels.JPG)

![Models](Images/Automl/BestModel.JPG)

![Models](Images/Automl/HyperParameters.JPG)

### Primary Parameter (Best AutoML Model)
Accuracy: 72.1%
![Models](Images/Automl/SummaryGraph.JPG)

### Feature Evaluation

When elements of the data were examined it was found that a single feature was found to have an outsized impact on the likelyhood of a customer purchase. 

![Importance](Images/Automl/FeatureImportance.JPG)


## Hyperparameter Tuning
The model choosen for the hyperparameter run is Logistic Regression. In order to setup the run there were several tasks that need to be performed. 

Setup the following 
* RandomParameterSampling setup:  This was selected because it would allow the range of values selected for the run.  It also supports early termination of a runs that are underperforming. 
* BanditPolicy Setup :  Bandit terminates any runs where the primary metric is not within the specified slack factor compared to the best performing run.


max_concurrent_runs (4): The maximum number of runs to execute concurrently.
max_total_runs (21): The maximum total number of runs to create. This is the upper bound; there may be fewer runs when the sample space is smaller than this value.


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

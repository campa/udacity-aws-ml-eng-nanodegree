# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Stefano Emilio Campanini

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
What I learnt was the need to clenaup from the predictions the negative values. 
We are predicting a 'count' so it has to be zero or positive.

### What was the top ranked model that performed?
Autogluon best model was WeightedEnsemble_L3 , Kaggle score was 1.75994 ( Root Mean Squared Logarithmic Error (RMSLE) ).

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
I added the hoour of the day, extracted from the date-time of the sample. In the EDA , in the histogram, is possible to see that the count is almost constant during the day with the exception of 4 moments in the day.  Keeping the hours separated from the date-time helps the model to find this correlation.
Another hint to the models traning/selection for AutoGluon has been the use of categorical type for some features, in this way we were adding hints for AutoGluon to treat these as category and not an amount.
Also the feature _casual_ and _registered_ was dropped.

### How much better did your model preform after adding additional features and why do you think that is?
Autogluon best model was WeightedEnsemble_L3 , Kaggle score was 0.7775 Root Mean Squared Logarithmic Error (RMSLE)

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
My knowledge of the models used by Autogluon is limited, not easy for me to decide what to do. 


### If you were given more time with this dataset, where do you think you would spend more time?
One thing try to set Autogluon to use the exact metric of Kaggle RMSLE, as default Autogluon was using RMSE not RMSLE. 
I suppose is possible to configure it as custom metric.
Another thing I would try is to check and think better using EDA in order to find some hints for features selection o engineering. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation

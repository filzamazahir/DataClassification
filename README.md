# Trees of Predictors - Data Classification Project
This project compares three existing tree ensembling methods for data classification, and implements a new one called Trees of Predictors (ToPs). The three existing ones are Random Forest Classifier, Extremely Randomized Trees Classifier, and AdaBoost Classifier.

## To Run

1) Clone the project
```
git clone https://github.com/filzamazahir/TreesOfPredictors-DataClassification.git
```
Make sure to have pip installed, then do 
```
pip install requirements.txt
```
Run dataclassification_project_ece657a.py

## Results
The new ToPs algorithm implemented had the lowest logarithmic loss and highest AUC-ROC and accuracy compared to the other classification method. However it did very poorly in terms of training time as it is very computationally intensive. It was also noticed that ToPs did not have a good stopping criterion which leads to overfitting.

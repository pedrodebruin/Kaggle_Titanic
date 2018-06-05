# Kaggle_Titanic
Repo for Titanic tutorial competition on Kaggle

## Analysis outline

All the data preprocessing and analysis is performed with python (pandas, numpy) and scikit. 
The training data is preprocessed as outlined below.

## Data Files

All relevant files and folders, including the output predictions, are located in the _data_ folder.
The list of files are:

- __train.csv__: to be used to build your machine learning models.
- __test.csv__: to be used to see how well your model performs on unseen data. For the test set, the ground truth for each passenger is not provided. 
- <b>gender_submission.csv</b>: a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
- __prediction.csv__: prediction results on test set using simple logistic regression
- <b>prediction\_nn.csv</b>: prediction results on test set using a not-too-fancy neural network

## Data Dictionary
| Variable	| Definition			| Key 				|
|---		|---				|---				|
| survival 	| Survival 			| 0 = No, 1 = Yes		|
| pclass 	| Ticket class 			| 1 = 1st, 2 = 2nd, 3 = 3rd	|
| sex 		| Sex 				|				|
| Age 		| Age in years 			|				|
| sibsp 	| # of siblings / spouses aboard the Titanic 	|		|
| parch 	| # of parents / children aboard the Titanic 	|		|
| ticket 	| Ticket number 		|				|
| fare 		| Passenger fare 		|				|
| cabin 	| Cabin number 			|				|
| embarked 	| Port of Embarkation 		| C = Cherbourg, Q = Queenstown, S = Southampton |

## Data preprocessing

- __x-validation__: We set aside 10% of the training set to be used for cross-validation using PassengerId modulo.
- __numeric gender conversion__: 'male' and 'female' labels are converted to 1 and 0, respectively.

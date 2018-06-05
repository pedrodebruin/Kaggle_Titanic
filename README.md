# Titanic Machine Learning Kaggle Competition
Repo for Titanic tutorial competition on Kaggle

## Analysis outline

All the data preprocessing and analysis is performed with python (pandas, numpy) and scikit. 
The training data is preprocessed as outlined below.

Due to the simplicity of this particular exercise, all the analysis is contained in the file _titanic.py_.

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

After the initial preprocessing, all fits are performed in a scikit pipe that is preceded by a StandardScaler(), which
scales both mean and std dev of all features passed to the algorithm. The exact transformations can be seen explicitly
in the _titanic.py_ script, but we summarize most of them below:

- __x-validation__: We set aside 10% of the training set to be used for cross-validation using PassengerId modulo.
- __numeric gender conversion__: The 'male' and 'female' labels are converted to 1 and 0, respectively.
- __ticket strings__: The values for ticket strings are often a combination of a literal first part followed by a number. 
We have initially decided to drop the first word and replace the feature with only the numerical value of the 2nd word.
- __Cabin feature__: The cabin feature has a high amount of NA entries. Initially dropped this feature since it is also string-valued, but
could perhaps be replaced with unique numbers (e.g. 10^LETTER(idx) + cabin #).
- __Passenger Name__: We have initially dropped the passenger name feature as we assume families will often occupy the same cabin, and the sibsp/parch features.
should hopefully encapsulate the effect of having additional family aboard.
- __Age__: The passenger age feature has a non-negligible fraction of NA, which are replaced with the mean of the training set age distribution.
- __PassengerId__: Due to this being a purely indexing feature, we drop it from the training dataframe.

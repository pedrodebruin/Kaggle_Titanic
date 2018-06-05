# Render our plots inline in IPython
#%matplotlib inline

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import re
import string

# This will be useful to convert letter to ints
from string import ascii_lowercase, ascii_uppercase


def main():
	# Load the data
	train_df = pd.read_csv('./data/train.csv')
	test_df = pd.read_csv('./data/test.csv')

	# Set 10% of training data for cross-validation
	xval_df  = train_df[ (train_df.PassengerId%10 == 0) ] 
	train_df = train_df[ (train_df.PassengerId%10 != 0) ] 
	
	print("\nBeginning the pre-processing of the data\n")

#	print("\nParch array, what does it look like:\n{0}".format(train_df.Parch[1:50]))
#	print("Parch array data type is {0}".format(type(train_df.Parch[1])))

	print("Converting gender strings to binary integers")
	sex = {'male': 1,'female': 2}	# creating a dict file 
	genderconversion(train_df,sex)
	genderconversion(xval_df,sex)
	genderconversion(test_df,sex)
	
	print("Replacing ticket strings with integers and dropping initial string when present")
	ticketcleanup(train_df)
	ticketcleanup(xval_df)
	ticketcleanup(test_df)
	
	print("Convert embarkment city code to an int")
	train_df.Embarked = [ letterToInt(l) for l in train_df.Embarked ]
	xval_df.Embarked = [ letterToInt(l) for l in xval_df.Embarked ]
	test_df.Embarked  = [ letterToInt(l) for l in test_df.Embarked ]
	
	# Save a few more general variables
	n_train = len(train_df)
	n_train_survived = len(train_df[train_df.Survived==1])
	n_train_deceased = len(train_df[train_df.Survived==0])
	n_train_male = len(train_df[train_df.Sex==1])
	n_train_female = len(train_df[train_df.Sex==2])
	
	print ("\nBefore we train a logistic regression classifier, let's review our data\n")
	
	print("Head of train set:")
	print(train_df[:10])
	
	print("\nTrain data description:")
	print(train_df.describe())
	
	print("\nTrain data columns:")
	print(train_df.columns)
	print('')
	
	# Plot a few numeric distributions
	plt.subplot(1,2,1)
	plt.hist(train_df['Age'].dropna())
	plt.ylabel('Age')
	
	plt.subplot(1,2,2)
	plt.hist(train_df['Fare'])
	plt.ylabel('Fare')
	
	plt.show()
	
	# Just reading off from table for practice
	print( "Number of men: {0}".format(len(train_df[train_df.Sex==1])) )
	print( "Number of women: {0}".format(len(train_df[train_df.Sex==2])) )
	print( "Fraction of men that survived: {0:.2f}"  .format( float(len(train_df[ (train_df['Survived']==1) & (train_df['Sex']==1)   ])) / ( n_train_male  ) ) )
	print( "Fraction of women that survived: {0:.2f}".format( float(len(train_df[ (train_df['Survived']==1) & (train_df['Sex']==2) ])) / ( n_train_female) ) )
	
	
	print("\nThere are too many missing entries for Cabin, let's not use it")
	train_df = train_df.drop(columns=['Cabin'])
	xval_df = xval_df.drop(columns=['Cabin'])
	test_df = test_df.drop(columns=['Cabin'])

	print("\nLet's drop the name of the passenger to assume it plays no role in their survival")
	train_df = train_df.drop(columns=['Name'])
	xval_df = xval_df.drop(columns=['Name'])
	test_df = test_df.drop(columns=['Name'])

	print("\nDropping PassengerId since that's just an indexing feature")
	train_df = train_df.drop(columns=['PassengerId'])
	xval_df  = xval_df.drop( columns=['PassengerId'])
	
	print("About {0:.2f}% of the data is missing values for 'Age'. Let's replace those missing values with the age mean".format(float(train_df['Age'].isnull().sum()*100./n_train)) )
	train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean()) # pandas automatically ignores NA/NaN when usingi aggregate functions like count() and mean()
	xval_df['Age'] = xval_df['Age'].fillna(xval_df['Age'].mean()) # pandas automatically ignores NA/NaN when usingi aggregate functions like count() and mean()
	test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()) # pandas automatically ignores NA/NaN when usingi aggregate functions like count() and mean()
	print("After the fix, {0:.2f}% of the data is missing values for 'Age', having been replaced with {1:.2f}.".format(float(train_df['Age'].isnull().sum()*100./n_train), train_df['Age'].mean() ) )

	# There's one random Fare entry with a missing value so I quickly set it to the mean:
	test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean()) 

	# Let's check our dataframes look good now	
	print("\nNumber of entries missing in train data: \n{0}".format(train_df.isnull().sum()))
	print("\nNumber of entries missing in xval data: \n{0}".format(xval_df.isnull().sum()))
	print("\nNumber of entries missing in test data: \n{0}".format(test_df.isnull().sum()))

	print ("#"*100)
	print ("\nEND OF PRE-PROCESSING OF DATA\n")
	print ("#"*100)
	
	
	# Now onto scikit learn
	from sklearn import linear_model
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LogisticRegression
	from sklearn.linear_model.logistic import _logistic_loss
	from sklearn.metrics import classification_report,confusion_matrix
	
	# For easier handling into scikit, let's put the Survived column in its own dataframe and make a new df with it dropped
	train_x_df = train_df
	train_x_df = train_x_df.drop(columns=['Survived'])
	train_y_df = train_df['Survived']
	xval_x_df = xval_df
	xval_x_df = xval_x_df.drop(columns=['Survived'])
	xval_y_df = xval_df['Survived']
	test_x_df = test_df
	test_x_df = test_x_df.drop(columns=['PassengerId'])	

	# Some of the Fare entries are strings. 
	dtypeCount_x =[train_x_df.iloc[:,i].apply(type).value_counts() for i in range(train_x_df.shape[1])]
	print(dtypeCount_x)

	
	print("\nWe start using a logistic classifier")
	pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
	
	# fit
	pipe.fit( train_x_df, train_y_df )

	
	print( "\nSummary of test dataframe:\n {0}".format(test_df.describe()) )
	
	# Split test dataset into cross-validation and test by using indexing column PassengerId

	# predict on xval sample to optimize method and hyperparameters
	xval_results = pipe.predict_proba(xval_x_df)
	#print("\nPrediction on xval sample is:\n")
	#print (xval_results)

	## Cost function
	#Jtrain =  _logistic_loss(clf.coef_, train_x_df, train_y_df, 1 / clf.C)
	#print("The cost as evaluated on the training set is:\n".format(Jtrain))	
	#Jtrain =  _logistic_loss(clf.coef_, train_x_df, train_y_df, 1 / clf.C)
	#print("The cost as evaluated on the cross-validation set is:\n".format(Jxval))	

	# predict on test to evaluate performance
	test_results = pipe.predict_proba(test_x_df)
	
	# Convert m x 2 probability array into a m x 1 logical 0 or 1 array based on index of max(p)
	test_logic_results = probToPred(test_results)
	print("Results on test set:\n{0}".format(test_results))
	
	print(len(test_logic_results))
	print (len(test_df['PassengerId']))

	pred_df = pd.DataFrame( { 'PassengerId': test_df['PassengerId'], 'Survived': test_logic_results })
	print(pred_df)
	pred_df.to_csv("data/prediction.csv", index=False)

	print("\n")
	print("#"*100)
	print("Now let's try a neural network")
	print("#"*100)

	from sklearn.neural_network import MLPClassifier

	pipe_nn = make_pipeline(StandardScaler(), 
		MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,10), random_state=1, ) )

	pipe_nn.fit( train_x_df, train_y_df )
	train_results = pipe.predict_proba(train_x_df)
	xval_results = pipe.predict_proba(xval_x_df)
	test_results = pipe_nn.predict_proba(test_x_df)

	train_logic_results = probToPred(train_results)
	xval_logic_results = probToPred(xval_results)
	test_logic_results = probToPred(test_results)

	print(classification_report(train_logic_results, train_y_df))
	print(classification_report(xval_logic_results, xval_y_df))

	pred_df = pd.DataFrame( { 'PassengerId': test_df['PassengerId'], 'Survived': test_logic_results })
	pred_df.to_csv("data/prediction_nn.csv", index=False)
 

def ticketcleanup(df):
	tcol = []
	for i,row in df.iterrows():
	        ticket_cidx = df.columns.get_loc('Ticket')
	        t_original = df.Ticket[i]

		# There a weird entry with value "LINE"...
		if t_original == "LINE":
			tnew = -1
		else:
			re.sub('[^A-Za-z0-9]+', '', t_original) # drop all punctuation
	        	t = t_original.split()
#			print("Converted {0} to {1}".format(t_original, t[-1]))
	        	tnew = int(t[-1])
	        #df.Ticket[i, ticket_cidx] = tnew
	        tcol.append(tnew)

	df.Ticket = tcol


def genderconversion(df,sexdict):
	
	# traversing through dataframe gender column and writing values where key matches
	df.Sex = [sexdict[item] for item in df.Sex]


def letterToInt(l):

	letters = list(string.ascii_lowercase)
	LETTERS = list(string.ascii_uppercase)
	n = 0
	try:
		n = LETTERS.index(l)
	except:
		print("Couldn't convert {0} to an int".format(l))

#	print ("Converted {0} to {1}".format(l,n))
	return n
	

def probToPred(df):
	newdf = []
	for i in range(0,len(df)):
		newdf.append(np.argmax(df[i]))	

	return newdf

if __name__=='__main__':
	main()

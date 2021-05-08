# Supervised machine learning models for dementia prediction

## Background and motivation
Dementia is a collective term associated with cognitive decline or other thinking skills severe enough to reduce a person’s ability to perform everyday activities. World wide, there are around 50 million cases presently, with an average of 10 million new cases each year. Early detection of dementia could really help keep the symptoms at bay and live a better life. Thus, early detection is highly important, which is difficult to perform manually on such a large scale. However, automated models can truly help boost the process.

## Project description
a) Creating different supervised machine learning models for predicting dementia<br>
b) Comparing the above models to find the best among them

## Data set description
Data source : https://www.oasis-brains.org/ <br>
Data set 	: OASIS 2
* Longitudinal data
* 150 subjects spanning 373 records for MRI scans over multiple visits- aged 60 to 96
* MRI scans for 2 or more visits - at least one year apart
* Subjects - right-handed, gender - M&F
* Records classification - 190 nondemented  and 146 demented
* 37 records - converted from nondemented to demented
* Features planned to use initially (from intuition):   
	- Age 
	- Gender
	- SES  		affects the diet and livelihood of subject
	- MMSE  	30 point questionnaire used to test subjects 
	- CDR  		5-point scale to characterize domains of cognitive and functional performance
	- eTIV  	provides a approximation of maximum morbid brain volume
	- nWBV  	normalized volume of the entire brain
* note: all the modifications have been done within the code only. No external data set modeificaiton has been performed. 
## Pre-reqs
a) tool for python programming - we've used Jupyter notebook (recommended) <br>
b) packages:<br>
	- pandas			for importing dataset <br>
	- seaborn			data exploration <br>
	- sklearn			model building and prediciton <br>
	- matplotlib		visualizations <br>
	
## Running the code
1) make sure to have the '.csv' data file in the right folder as in code
2) if all packages are installed already, run the code as is, the orede has been set.
3) Use the model to predict new data, if needed
4) model equation has been given for ease of use - can be performed manually as well.

## Section-wise code:
1) Data preparation
	- importng data set
	- removing redundant data
	- handling null / missing values
	- categorical data to numeric data
2) Data exploration
	- distribution of age
	- distribution of education 
	- distribution of gender
	- pairplot
3) Feature selection
	- correlation table
	- sorting correlation values
4) Model building 
	- helper function for building 6 different models
		: fitting data and 5-fold cross validation
	- model for different feature combinaitons
		: all
		: top 5
		: top 4
		: top 3
	- comparing the above models base don accuracy from cross validation
	- plotting and tabulating model accuracies
	- total comparisions = 6 (models) * 4 (feature combos) = 24
5) Model evaluation
	- helper funcions <br>
		: plotting confusion matrix <br>
		: printing report <br>
		: plotting ROC curve with AUC <br>
		: model building and evaluation using metrics <br>
	- calculating and plotting metrics for each model 
	- total comparisions = 6 (models) with top 4 features in each
6) Prediction
	- using the best model from above analysis (Logistic Regression)
	- predicting values for test data set
	- calculating total numver of wrongly predicted rows
7) Model equation
	- creating model equation from intercept and coefficients from model<br>
	<strong> Y = 11.408 + 5.5268(x1) - 0.437(x2) - 0.309(x3) - 0.235(x4)<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where, <br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Y  = Group number<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x1 = CDR <br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x2 = MMSE<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x3 = nWBV<br>
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x4 = M/F </strong>
	
## References:
- https://www.oasis-brains.org/#oasis2
- https://www.who.int/news-room/fact-sheets/detail/dementia
- https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html

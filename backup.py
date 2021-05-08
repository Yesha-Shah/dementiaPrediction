#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
from statsmodels.graphics.mosaicplot import mosaic
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# <h2> Data preparation </h2>

# In[2]:


#importing dataset
df = pd.read_csv('oasis_longitudinal.csv')


# In[3]:


df['Group'].value_counts()


# In[4]:


df.tail()


# In[5]:


#Removing redundant data
df = df.drop(['Hand', 'MR Delay', 'Visit', 'Subject ID', 'MRI ID'], axis = 1)
df.tail()


# In[6]:


df.describe()


# In[7]:


#checking for null values
df.isna().sum()


# In[8]:


#dropping null values from MMSE
df = df.dropna(subset=['MMSE'])


# In[9]:


sns.distplot(df['SES'], hist = True)


# In[10]:


sns.distplot(df["SES"].fillna(df["SES"].mean()), hist=True)


# In[11]:


sns.distplot(df["SES"].fillna(df["SES"].median()), hist=True)


# In[12]:


#replacing null values with mean so as to avoid change in distribution
df["SES"].fillna(df["SES"].mean(), inplace=True)


# In[13]:


#confirming no more null values
df.isna().sum()


# In[14]:


#Replacing 'Converted' group as 'Demented'
df['Group'] = df['Group'].replace('Converted', 'Demented')


# In[ ]:





# In[15]:


#converting categorical to numeric data
df['M/F'] = df['M/F'].replace('M', '0')
df['M/F'] = df['M/F'].replace('F', '1')

df['GroupNum'] = df['Group']
df['GroupNum'] = df['GroupNum'].replace('Nondemented', '0')
df['GroupNum'] = df['GroupNum'].replace('Demented', '1')


# In[16]:


#converting data type from object to int
df['GroupNum'] = pd.to_numeric(df['GroupNum'])
df['M/F'] = pd.to_numeric(df['M/F'])


# In[17]:


df.info()


# In[18]:


df.tail()


# <h2> Data exploration </h2>

# In[19]:


#distribution of age
sns.distplot(df['Age'], bins = 10, hist = True)


# In[20]:


#distribution of educaiton
sns.distplot(df['EDUC'], hist = True)


# In[21]:


#distribution by gender
mos = mosaic(df, ['M/F', 'Group'])


# In[22]:


sns.pairplot(df, hue = 'Group', palette='husl')


# <h2> Feature selection </h2>

# In[23]:


corr = df.corr()
corr.style.background_gradient(cmap='PRGn')


# In[24]:


pd.DataFrame(abs(corr['GroupNum']).sort_values(ascending=False)).rename(columns={'GroupNum' : 'Correlation with target'})


# <h3> Features- highest to lowest correlation with group</h3>
# <ul><li>CDR</li>
#     <li>MMSE</li>
#     <li>nWBV</li>
#     <li>M/F</li>
#     <li>EDUC</li>
#     <li>SES</li>
#     <li>eTIV</li>
#     <li>ASF</li>
#     <li>Age</li>

# <h2> Model building </h2>

# In[25]:


#models under consideration

models = ['Logistic Regression',
           'Random Forest',
           'Support Vector Machine',
           'KNeighbors',
           'Decision Tree',
           'XGBoost']


# In[26]:


def compareModels(X_train, X_test, y_train, y_test):
    #building models
    LR = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=555)
    RF = RandomForestClassifier(n_estimators = 100, random_state=555)
    SVM = SVC(random_state=0, probability=True)
    KNC = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    clf_XGB = XGBClassifier(n_estimators = 100, seed=555, use_label_encoder=False, eval_metric='logloss')

    trainAccuracies = []
    testAccuracies = []
    print('5-fold cross validation:\n')
    for clf, label in zip([LR, RF, SVM, KNC, DTC, clf_XGB],
                          models):
        #fitting the model
        md = clf.fit(X_train, y_train)
        
        #cross-validation scores (accuracies)
        scores = sklearn.model_selection.cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        print("Train CV Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
        trainAccuracies.append(scores.mean())
        testAcc = sklearn.metrics.accuracy_score(clf.predict(X_test), y_test)
        testAccuracies.append(testAcc)
        print("Test Accuracy: %0.4f " % (testAcc))
        
    return trainAccuracies, testAccuracies


# In[27]:


#model with all features from data set

X = df.drop(columns=['GroupNum', 'Group'])
y = df['GroupNum']
print('For all features')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=50)
trainAccurAll, testAccurAll = compareModels(X_train, X_test, y_train, y_test)


# In[28]:


#model with top 5 features

X = df[['CDR','MMSE', 'nWBV', 'M/F', 'EDUC']]
y = df['GroupNum']
print('For top 5 correlated features')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=50)
trainAccur5Feat, testAccur5Feat = compareModels(X_train, X_test, y_train, y_test)


# In[29]:


#model with top 4 features

X = df[['CDR','MMSE', 'nWBV', 'M/F']]
y = df['GroupNum']
print('For top 4 correlated features')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=50)
trainAccur4Feat, testAccur4Feat = compareModels(X_train, X_test, y_train, y_test)


# In[30]:


#model with top 3 features

X = df[['CDR','MMSE', 'nWBV']]
y = df['GroupNum']
print('For top 3 correlated features')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=50)
trainAccur3Feat, testAccur3Feat = compareModels(X_train, X_test, y_train, y_test)


# In[31]:


#comparing models with different features based on accuracies

fig, ax = plt.subplots(2, 2, figsize = (15, 15))
fig.tight_layout(pad=15)

ax[0, 0].scatter(models, trainAccurAll, label = 'Train')
ax[0, 0].scatter(models, testAccurAll, label = 'Test')
ax[0, 0].set_xticklabels(rotation = 60, labels = models)
ax[0, 0].set_title('All features \n [CDR, MMSE, nWBV, M/F, EDUC, SES, eTIV, ASF, Age]')

ax[0, 1].scatter(models, trainAccur5Feat)
ax[0, 1].scatter(models, testAccur5Feat)
ax[0, 1].set_xticklabels(rotation = 60, labels = models)
ax[0, 1].set_title('Top 5 features \n [CDR, MMSE, nWBV, M/F, EDUC]')
ax[0, 1].set_ylim(bottom=0.5)

ax[1, 0].scatter(models, trainAccur4Feat)
ax[1, 0].scatter(models, testAccur4Feat)
ax[1, 0].set_xticklabels(rotation = 60, labels = models)
ax[1, 0].set_title('Top 4 features \n [CDR, MMSE, nWBV, M/F]')
ax[1, 0].set_ylim(bottom=0.5)

ax[1, 1].scatter(models, trainAccur3Feat)
ax[1, 1].scatter(models, testAccur3Feat)
ax[1, 1].set_xticklabels(rotation = 60, labels = models)
ax[1, 1].set_title('Top 3 features \n [CDR, MMSE, nWBV]')
ax[1, 1].set_ylim(bottom=0.5)

fig.legend()


# In[32]:


#Test accuracy comparision

compareModels = pd.DataFrame()
compareModels['All features test'] = testAccurAll
compareModels['5 features test'] = testAccur5Feat
compareModels['4 features test'] = testAccur4Feat
compareModels['3 features test'] = testAccur3Feat
compareModels.index = models
compareModels.style.background_gradient(cmap='PRGn')


# In[33]:


#Train accuracy comparision

compareModelsTrain = pd.DataFrame()
compareModelsTrain['All features train'] = trainAccurAll
compareModelsTrain['5 features train'] = trainAccur5Feat
compareModelsTrain['4 features train'] = trainAccur4Feat
compareModelsTrain['3 features train'] = trainAccur3Feat
compareModelsTrain.index = models
compareModelsTrain.style.background_gradient(cmap='PRGn')


# In[ ]:





# <h2> Model evaluation </h2>

# In[34]:


#helper function to plot confusion matrix

def plot_confusion_matrix(cm):
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.winter)
    classNames = ['Nondemented','Demented']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[35]:


#helper function

def printReport(report):
    values = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    for i in range(4):
        print('\t', values[i], ' = ', round(report[i], 4))
    print('\n')


# In[36]:


#helper function for plotting rocCurve

def plotRoc_curves(fpr, tpr, thresholds):
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='seagreen', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[37]:


#model evaluation

def getMetrics(X_train, X_test, y_train, y_test):
    #model building
    LR = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=555)
    RF = RandomForestClassifier(n_estimators = 100, random_state=555)
    SVM = SVC(random_state=0, probability=True)
    KNC = KNeighborsClassifier()
    DTC = DecisionTreeClassifier()
    clf_XGB = XGBClassifier(n_estimators = 100, seed=555, use_label_encoder=False, eval_metric='logloss')

    confusionMatrices = []
    classificationReports = []
    predictions = []
    
    for clf, label in zip([LR, RF, SVM, KNC, DTC, clf_XGB],
                          models):
        #training the model
        md = clf.fit(X_train, y_train)
        #predicted values
        prediction = md.predict(X_test)
        
        predictions.append(prediction)
        
        #confusion matrix
        cm = metrics.confusion_matrix(y_test, prediction)
        confusionMatrices.append(cm)
        
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        tot = TN + FP + FN + TP
        
        #calculating model metrics
        accuracy =  (TP + TN) / tot
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1Score = 2*(recall * precision) / (recall + precision)
        classificationReports.append([accuracy, precision, recall, f1Score])
        
    return confusionMatrices, classificationReports, predictions


# In[38]:


#Selecting model with top 4 features and spliting data into train and test

X = df[['CDR','MMSE', 'nWBV', 'M/F']]
y = df['GroupNum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=50)
confusionMatrices, classificationReports, predictions = getMetrics(X_train, X_test, y_train, y_test)


# In[39]:


#metrics for individual models

print('\t=======\033[1mLogistic Regression\033[0m=======')
plot_confusion_matrix(confusionMatrices[0])
printReport(classificationReports[0])
fpr, tpr, thresholds = roc_curve(predictions[0], y_test)
plotRoc_curves(fpr, tpr, thresholds)


print('\t=======\033[1mRandom Forest Classifier\033[0m=======')
plot_confusion_matrix(confusionMatrices[1])
printReport(classificationReports[1])
fpr, tpr, thresholds = roc_curve(predictions[1], y_test)
plotRoc_curves(fpr, tpr, thresholds)

print('\t\t=======\033[1mSVM\033[0m=======')
plot_confusion_matrix(confusionMatrices[2])
printReport(classificationReports[2])
fpr, tpr, thresholds = roc_curve(predictions[2], y_test)
plotRoc_curves(fpr, tpr, thresholds)

print('\t=======\033[1mK-Neighbors Classifier\033[0m=======')
plot_confusion_matrix(confusionMatrices[3])
printReport(classificationReports[3])
fpr, tpr, thresholds = roc_curve(predictions[3], y_test)
plotRoc_curves(fpr, tpr, thresholds)

print('\t\t=======\033[1mDecision Tree\033[0m=======')
plot_confusion_matrix(confusionMatrices[4])
printReport(classificationReports[4])
fpr, tpr, thresholds = roc_curve(predictions[4], y_test)
plotRoc_curves(fpr, tpr, thresholds)

print('\t\t=======\033[1mXG Boost\033[0m=======')
plot_confusion_matrix(confusionMatrices[5])
printReport(classificationReports[5])
fpr, tpr, thresholds = roc_curve(predictions[5], y_test)
plotRoc_curves(fpr, tpr, thresholds)


# <h2> Prediction </h2>
# <h5> using the best model based on above analysis </h5>

# In[40]:


LR = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=555)
md = LR.fit(X_train, y_train)
prediction = pd.DataFrame(md.predict(X_test))
print("Test set accuracy: {:.2f}".format(LR.score(X_test, y_test)))


# In[41]:


prediction['Actual'] = pd.DataFrame(y_test).reset_index().drop('index', axis=1)['GroupNum']
prediction.rename(columns={0 : 'Predicted'}, inplace=True)
prediction
#0 = non-demented
#1 = demented


# In[42]:


#rows with wrong predictions (total = 5)

prediction[prediction['Predicted']!=prediction['Actual']]


# In[43]:


print('Intercept= ', LR.intercept_)
print('Coefficients = ', *LR.coef_)


# <h3> Model equation </h3><br>
# Y = 11.408 + 5.5268(x1) - 0.437(x2) - 0.309(x3) - 0.235(x4)
# 
# where, <br>
# Y &nbsp; = Group number<br>
# x1 = CDR<br>
# x2 = MMSE<br>
# x3 = nWBV<br>
# x4 = M/F

# In[ ]:





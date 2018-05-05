import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import seaborn as sns
from scipy import stats
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

plt.rc("font", size=14)
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data_features = pd.read_csv("features.csv",header=0)
data_label = pd.read_csv("labels.csv",header=0)
data = pd.merge(data_features, data_label, on='ID')
data = data.dropna()

#To know different statistics about data

print(data.shape)
print(list(data.columns))
print(data.head())
print(data['Sickness'].value_counts())

#Plot total sick people and plot

sns.countplot(x='Sickness', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

#Taking mean against Sickness

print(data.groupby('Sickness').mean())

#Plotting one of the feature

pd.crosstab(data.Feature1,data.Sickness).plot(kind='bar')
plt.title('Frequency for Feature1')
plt.xlabel('Feature1')
plt.ylabel('Frequency of Feature1')
plt.show()
plt.savefig('Frequency of Feature1')


#Feature Selection

data_final_vars=data.columns.values.tolist()
y=['Sickness','ID']
Y=['Sickness']
X=[i for i in data_final_vars if i not in y]
print(X,y)

logreg = LogisticRegression()


rfe = RFE(logreg, 20)
rfe = rfe.fit(data[X], data[Y] )
print(rfe.support_)
print(rfe.ranking_)



cols = ["Feature15","Feature23","Feature43","Feature45","Feature64",
"Feature87","Feature115","Feature127","Feature162","Feature163",
"Feature236","Feature362","Feature379","Feature442","Feature451",
"Feature452","Feature495","Feature551","Feature634","Feature655",]

X=data[cols]
y=data['Sickness']

#Get the summary against each Feature

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Training the model

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
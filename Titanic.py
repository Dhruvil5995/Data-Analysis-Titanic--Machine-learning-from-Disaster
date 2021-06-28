import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data= pd.read_csv('../input/titanic/gender_submission.csv')
data.head()

train= pd.read_csv('../input/titanic/train.csv')
test= pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()

train.isnull().sum()
#177 rows with age are missing, 687 rows(information) missing with cabin and 2 with embarked information 

test.isnull().sum()
##86 rows with age are missing,327 rows(information) missing with cabin and 1 with Fare information

print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())

def fig(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead= train[train['Survived']==0][feature].value_counts()
    a= pd.DataFrame([survived,dead])
    a.index=['Survived','Dead']
    a.plot(kind='bar',stacked=True, figsize=(10,5))

fig('Sex')
fig('Pclass')

train_test_data= [train,test]
# combining training and test dataset

for dataset in train_test_data:
    dataset['Titel']= dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        

train['Titel'].value_counts()  
 
titel_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,
              "Col":3,"Mlle":3,"Major":3,"Mme":3,
              "Lady":3,"Don":3,"Sir":3,"Ms":3,"Jonkheer":3,"Capt":3,"Countess":3 }
​

for dataset in train_test_data:
   dataset['Titel']= dataset['Titel'].map(titel_mapping)
​
#delete unnecessary information from dataset
train.drop(['Name','Ticket','Cabin','Embarked','SibSp','Parch','Titel'] ,axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin','Embarked','SibSp','Parch','Titel'] ,axis=1,inplace=True)

sex_mapping={"male":1, "female":0}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)

#fill missing data of Age and fare
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)

train['Fare'].fillna(train['Fare'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)


#Add feature scalling
sc= StandardScaler()

feature_scalling=['Age','Fare']
train[feature_scalling]=sc.fit_transform(train[feature_scalling])

feature_scalling=['Age','Fare']
test[feature_scalling]=sc.fit_transform(test[feature_scalling])

x=train.drop(['Survived'],axis=1)
y=train['Survived']

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.svm import SVC


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, x, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)

sc= SVC(C=100,kernel='rbf')
model= sc.fit(x,y)

y_pred=sc.predict(test)
print(y_pred)


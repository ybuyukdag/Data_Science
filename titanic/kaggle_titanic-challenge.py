import pandas as pd
import numpy as np


data_train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
combine = [data_train,test]

result = data_train.columns.values
result = data_train.head()
#result = data_train.info()
#result = test.info()
result = data_train.describe() #distribution of rate of numerical features
result = data_train.describe(include=[object]) #distibution of categorical features
#classifying
result = data_train[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False) #Classifying by ticket class and analyse the correlation with surviving
result = data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) #correlation of sex and survive
result = data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False) #correlation with people who were with his/her wife/husband or siblings and survive
result = data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False) #correlation with the people who were with parents or child and survive

#print(result)

#Graphical analyse
import matplotlib.pyplot as plt 
import seaborn as sns

#correlation of numerical features with survive
#AGE
'''
age_sur = sns.FacetGrid(data_train,col='Survived') #graphics about categorization survived (0/1) people by age range
age_sur.map(plt.hist, 'Age', bins=20)
plt.show()
'''
'''
pcl_sur = sns.FacetGrid(data_train, col='Survived', row='Pclass', aspect=1.6, legend_out=True) #correlation of age and survive based on pclass
pcl_sur.map(plt.hist, 'Age' , alpha = 0.5, bins=20)
plt.show()
'''
#correlation of categorical features 
#Sex,Embarked,Pclass
'''
emb_sur = sns.FacetGrid(data_train, row="Embarked", height=2.2, aspect=1.6)
emb_sur.map(sns.pointplot, data=data_train, x="Pclass",y="Survived", hue="Sex", order = [1,2,3], hue_order = ["male", "female"]) # females are survived more than males
emb_sur.add_legend()
plt.show()
'''
#Categorical and numerical feautures correlation
#Embarked,Sex,Fare
'''
fare_sur = sns.FacetGrid(data_train, row="Embarked", col="Survived", height = 2.2, aspect=1.6) #there is a correlation between embarked and survive 
fare_sur.map(sns.barplot, "Sex", "Fare", order= ["female","male"],alpha=0.5, errorbar=None) #more paid passengers have high survived rate.
fare_sur.add_legend
plt.show()
'''
#Wrangling data/Correcting,Creating,Completing
'''
print("Before", data_train.shape, test.shape,combine[0].shape, combine[1].shape) #Dropping some feature because they wont contribute to analyse. less data-more speed
data_train = data_train.drop(["Ticket","Cabin"], axis=1)
test = test.drop(["Ticket","Cabin"], axis=1)
combine = [data_train, test]
print("After", data_train.shape, test.shape,combine[0].shape, combine[1].shape)
'''
#NewFeaturesBeforeDrop

data_train["Title"] = data_train.Name.str.extract("([A-Za-z]+)\.", expand=False) #new column for title. Title can have correalation with survive
df = pd.crosstab(data_train["Title"],data_train["Sex"]) 
df = pd.crosstab(data_train["Title"],data_train["Survived"]) #survival based on titles

data_train["Title"] = data_train["Title"].replace(["Lady",'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') #categorizing title feature into small pieces
data_train["Title"] = data_train["Title"].replace("Mlle","Miss")
data_train["Title"] = data_train["Title"].replace("Ms","Miss")
data_train["Title"] = data_train["Title"].replace("Mme","Mrs")
df = data_train[["Title","Survived"]].groupby(["Title"], as_index=False).mean() #after combining data, grouping the average survival rate of title
#print(df)
##
data_train = data_train.drop(["Name", "PassengerId"], axis=1) #do not need this features anymore
test = test.drop(["Name"], axis=1)
combine = [data_train,test]
#print(data_train.shape,test.shape)

##
data_train["Sex"] = data_train["Sex"].map({"female":1,"male":0}).astype(int) #Sex feature is str->numerical now. This way help us reach completing
#print(data_train.head())

##Filling Numerical Features
'''
pcl_sex = sns.FacetGrid(data_train, row="Pclass", col="Sex", height=2.2, aspect=1.6)
pcl_sex.map(plt.hist,"Age", alpha=.5, bins=20)
pcl_sex.add_legend()
plt.show()
'''

guess_ages = np.zeros((2,3)) #filling ages by prediction based on correlation of pclass and sex
guess_ages = guess_ages
#df = data_train[(data_train["Sex"]==1) & (data_train["Pclass"]==1)]["Age"].dropna().median() #median of age
#print(df)

# for dataset in combine:
#     for i in range(0,2):
#         for j in range(0,3):
#             guess_df = dataset[(data_train["Sex"]==i) & (dataset["Pclass"]==j+1)]["Age"].dropna()
#             age_mean = guess_df.mean() #age average
#             age_std = guess_df.std() #standart deviation
#             age_guess = rnd.uniform(age_mean-age_std, age_mean+age_std)

#             age_guess = guess_df.median()
            
#             guess_ages[i,j] = (age_guess/0.5+0.5)* 0.5 #convert rndm age float to nearest 0.5 age

#     for i in range(0, 2):
#         for j in range(0, 3):
#             dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

#     data_train[['Age']] = data_train[['Age']].astype(float)
# #print(data_train.head())

## Filling Embarked feature
fill_port = data_train.Embarked.dropna().mode()[0]
#print(fill_port)

for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(fill_port)
df = data_train[["Embarked","Survived"]].groupby(["Embarked"], as_index=False).mean()
#print(df)

#Filling Fare feature at Test dataset
'''test.info()'''
test["Fare"].fillna(test["Fare"].dropna().median(),inplace =True)
#print(test.head())
#test.info()

#MODELLING, PREDICTING, SOLVING

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#dataset for validation
X = data_train.drop("Survived",axis=1)
y = data_train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

print(type(X_train), type(y_train), X_test.shape)

# # Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(acc_log)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print(acc_svc)

#k-Nearest Neigbours algoritm
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print(acc_gaussian)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print(acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print(acc_linear_svc)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print(acc_sgd)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print(acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest)

###Model Assesment
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

#submission
# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('submission.csv', index=False)


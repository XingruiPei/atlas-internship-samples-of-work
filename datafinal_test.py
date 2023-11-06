
"""
1. Objective of the coding process:
   - The primary objective of this code is to analyze and visualize the churn rate of a bank's customers.
     The analysis investigates how different variables (like age, gender, credit card ownership, and more)
     influence a customer's decision to leave (or churn) from the bank. Furthermore, the code also aims
     to build machine learning models to predict the churn based on various features.

2. Data:
   - The data represent customer records for a bank, capturing details like credit scores,
     geographical location, gender, age, tenure, account balance, number of products, credit card ownership,
     membership activity status, estimated salary, complaints, satisfaction scores, card types, points earned,
     and age categories.
   - The data is a 10,000 line csv file
   - The data is from a open source database

3. Packages/libraries:
   - The following Python libraries are used:
     - `pandas`: For data manipulation and analysis.
     - `matplotlib` and `seaborn`: For data visualization.
     - `sklearn`: Specifically, it uses modules like `OrdinalEncoder` for encoding categorical variables,
       `train_test_split` for splitting the dataset, and different model classes and metrics
       (like `XGBClassifier`, `accuracy_score`, `precision_score`, etc.) for building and evaluating machine learning models.
     - `xgboost`: For the `XGBClassifier` model, which is a gradient boosting framework.
     - `warnings`: To ignore specific warnings related to the seaborn library.

4. Functions:
   - The code employs various functions for its operations, and some notable ones include:
     - Data reading (`pd.read_csv`).
     - Data visualization functions (`sns.countplot`, `plt.subplots`, `plt.figure`, `sns.heatmap`, etc.).
     - Data processing functions (`pd.cut` for binning, `drop` for removing columns, etc.).
     - Machine learning related functions (`train_test_split` for data splitting, `fit` for training models,
       `predict` for making predictions, and various metric functions for model evaluation).
"""



##
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# Setting up font and ignoring seaborn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

plt.rcParams['font.sans-serif']=['Songti SC']  # Setting font

##

data = pd.read_csv("/Users/peixingrui/Desktop/Customer-Churn-Records.csv")  # Loading data

print(data.head())  # Displaying first few rows

##
print(data.shape)  # Displaying data shape
print(data.info())  # Displaying data info

##
# Removing columns 'RowNumber', 'CustomerId', 'Surname' as they are not relevant for analysis
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

##
# Visualizing user churn ratio
plt.pie(data["Exited"].value_counts(), labels=["Not Churned", "Churned"], autopct="%0.2f%%")
plt.show()

##
# Renaming columns for easier access
data.columns = ['CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited', 'Complain',
       'SatisfactionScore', 'CardType', 'PointEarned']

##
# Categorizing age into different age groups
age_bins=[0,11,25,45,float('inf')]
age_labels=['child','teenager','adult','old']
data['CategoryAge']=pd.cut(data['Age'],bins=age_bins, labels=age_labels)
print(data)

##
# Analyzing different age groups
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()
# Count of users in different age groups
sns.countplot(x='CategoryAge',data=data,ax=ax[0])
ax[0].set_title('User count by age group')
# Churn by different age groups
sns.countplot(x='CategoryAge',hue='Exited',data=data,ax=ax[1])
ax[1].set_title('Churn by age group')
plt.show()

## From the results, the bank's main customers are adults and the elderly. Notably, the elderly have a high churn rate.
# To understand this, we further analyze the elderly group to see the factors affecting their churn.
data_old = data[data['CategoryAge'] == 'old']
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
ax = ax.flatten()
for i, column in enumerate(
        ['Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Complain', 'SatisfactionScore',
         'CardType', 'Geography']):
    sns.countplot(x=column, hue='Exited', data=data_old, ax=ax[i])
    ax[i].set_title(f'Old churn by {column}', fontsize=15)
    ax[i].set_xlabel(column)
plt.tight_layout()
plt.show()

"""
Based on the provided graphs, here are some conclusions we can draw:

1. Churn by Gender: There's a higher churn rate among females compared to males.

2. Churn by Tenure: The churn rate appears relatively consistent across different tenures, but there seems to be a slight increase in churn for customers with 1 year of tenure.

3. Churn by Number of Products: Customers with 1 or 2 products have a lower churn rate than those with 3 products. There's very minimal churn for those with 4 products.

4. Churn by Credit Card: The churn rate is slightly higher for customers who have a credit card (HasCrCard = 1) than those who don't.

5. Churn by Membership Activity: Inactive members (IsActiveMember = 0) have a significantly higher churn rate compared to active members.

6. Churn by Complaints: Customers who have lodged complaints (Complain = 1) have a much higher churn rate than those who haven't.

7. Churn by Satisfaction Score: Churn rate seems to decrease as the satisfaction score increases, with a slight anomaly for score '4', which has a higher churn compared to score '3'.

8. Churn by Card Type: Customers with SILVER and DIAMOND card types have a higher churn rate compared to those with PLATINUM and GOLD card types.

9. Churn by Geography: The churn rate is highest in Germany, followed by Spain and then France.

In summary, factors such as gender, credit card ownership, membership activity, complaints, satisfaction score, card type, and geography play a role in the churn rate of customers. The data suggests that focusing on improving customer satisfaction, ensuring more active membership, addressing complaints, and targeting specific geographies might help in reducing customer churn.
"""

##
# Churn by different factors for adults

data_adult = data[data['CategoryAge'] == 'adult']
plt.figure(figsize=(20, 50))
count_plt = ['Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Complain', 'SatisfactionScore',
             'CardType', 'Geography']
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
ax = ax.flatten()
for i, column in enumerate(count_plt):
    sns.countplot(x=column, hue='Exited', data=data_adult, ax=ax[i])
    ax[i].set_title(f'Adult churn by {column}', fontsize=15)
    ax[i].set_xlabel(column)
plt.tight_layout()
plt.show()

#

##
# Churn by different factors for teenagers
data_tee = data[data['CategoryAge'] == 'teenager']
plt.figure(figsize=(20, 50))
count_plt = ['Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Complain', 'SatisfactionScore',
             'CardType', 'Geography']
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
ax = ax.flatten()
for i, column in enumerate(count_plt):
    sns.countplot(x=column, hue='Exited', data=data_adult, ax=ax[i])
    ax[i].set_title(f'Teenager churn by {column}', fontsize=15)
    ax[i].set_xlabel(column)
plt.tight_layout()
plt.show()

## Encoding categorical columns
cat_cols = ["Geography", "Gender", "CardType","CategoryAge"]
enc = OrdinalEncoder()
data[cat_cols] = enc.fit_transform(data[cat_cols])
print(data)

## Displaying correlation matrix
corr = data.corr().round(2)
plt.figure(figsize = (20,10))
sns.heatmap(corr, annot = True, cmap = 'YlOrBr')

"""
The values on the diagonal are all 1 because they represent the correlation between the same variables.
"Exited" (churn) and "Age" have a positive correlation of 0.29, implying that as age increases, 
the likelihood of churn also increases.
"Exited" and "IsActiveMember" (active membership status) have a negative correlation of -0.16, 
indicating that active members are less likely to churn.
"NumOfProducts" (number of products) and "Balance" (account balance) have a negative correlation of -0.3, 
suggesting that customers with more products typically have a lower account balance.
"Exited" and "Complain" (complaints) have a positive correlation of 0.16, 
meaning customers who have lodged complaints are more likely to churn.
"Category_Age" (age category) and "Age" have a positive correlation of 0.23, 
which is expected since they both relate to age.
Most variables have very low correlations with each other, 
indicating that there isn't a strong linear relationship between them.
"""

##
X_train, X_test, y_train, y_test = train_test_split(data.drop(["Exited","Age"], axis=1), data["Exited"], test_size=0.20, random_state=42)
# Building XGBClassifier model
model = XGBClassifier(
    n_estimators=750,
    max_depth=4,
    learning_rate=0.01,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="logloss",
    early_stopping_rounds=1000,
    verbose=100,
)
y_pred = model.predict(X_test)
print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
# Displaying evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("ROC AUC score:", roc_auc)

##
# Building Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression model:", accuracy)

##
# Building Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Random Forest model:", accuracy)

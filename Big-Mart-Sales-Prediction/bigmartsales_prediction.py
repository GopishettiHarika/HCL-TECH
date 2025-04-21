import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datasets
train = pd.read_csv(r"C:\Users\Admin\PycharmProjects\Big-Mart-Sales-Prediction\bm_Train.csv")
test = pd.read_csv(r"C:\Users\Admin\PycharmProjects\Big-Mart-Sales-Prediction\bm_Test.csv")

# making copies of train and test dataset
train = train.copy()
test = test.copy()

# Create a 2x4 grid for subplots (8 plots in total)
fig, axs = plt.subplots(2, 4, figsize=(16, 10))

# Plotting the first graph on the first subplot (axs[0, 0])
axs[0, 0].hist(train['Item_Outlet_Sales'], bins=20, color='pink')
axs[0, 0].set_title('Target Variable')
axs[0, 0].set_xlabel('Item Outlet Sales')
axs[0, 0].set_ylabel('Count')

# Plotting the second graph on the second subplot (axs[0, 1])
train['Item_Identifier'].value_counts().plot.hist(ax=axs[0, 1], color='blue')
axs[0, 1].set_title('Item Identifier Distribution')
axs[0, 1].set_xlabel('Item Identifier')
axs[0, 1].set_ylabel('Number of Items')

# Plotting the third graph on the third subplot (axs[0, 2])
train['Item_Fat_Content'].value_counts().plot.bar(ax=axs[0, 2], color='green')
axs[0, 2].set_title('Item Fat Content')
axs[0, 2].set_xlabel('Fat Content')
axs[0, 2].set_ylabel('Number of Items')

# Plotting the fourth graph on the fourth subplot (axs[0, 3])
train['Item_Type'].value_counts().plot.bar(ax=axs[0, 3], color='orange')
axs[0, 3].set_title('Item Type Distribution')
axs[0, 3].set_xlabel('Item Type')
axs[0, 3].set_ylabel('Number of Items')

# Plotting the fifth graph on the fifth subplot (axs[1, 0])
train['Outlet_Identifier'].value_counts().plot.bar(ax=axs[1, 0], color='purple')
axs[1, 0].set_title('Outlet Identifier Distribution')
axs[1, 0].set_xlabel('Outlet Identifier')
axs[1, 0].set_ylabel('Number of Items')

# Plotting the sixth graph on the sixth subplot (axs[1, 1])
train['Outlet_Size'].value_counts().plot.bar(ax=axs[1, 1], color='red')
axs[1, 1].set_title('Outlet Size Distribution')
axs[1, 1].set_xlabel('Outlet Size')
axs[1, 1].set_ylabel('Number of Items')

# Plotting the seventh graph on the seventh subplot (axs[1, 2])
train['Outlet_Location_Type'].value_counts().plot.bar(ax=axs[1, 2], color='brown')
axs[1, 2].set_title('Outlet Location Type')
axs[1, 2].set_xlabel('Location Type')
axs[1, 2].set_ylabel('Number of Items')

# Plotting the eighth graph on the eighth subplot (axs[1, 3])
train['Outlet_Type'].value_counts().plot.bar(ax=axs[1, 3], color='pink')
axs[1, 3].set_title('Outlet Type Distribution')
axs[1, 3].set_xlabel('Outlet Type')
axs[1, 3].set_ylabel('Number of Items')

# Adjust the layout to make sure plots do not overlap
plt.tight_layout()

# Add custom vertical spacing between rows
plt.subplots_adjust(hspace=0.5)  # Adjust this value for more or less space

# Show the combined plots
plt.show()

# Further Data Preprocessing and Model Steps follow

# checking unique values in the columns of train dataset
data = pd.concat([train, test])
data.apply(lambda x: len(x.unique()))

data.isnull().sum()

# imputing missing values
data['Item_Weight'] = data['Item_Weight'].replace(0, np.nan)
data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())

data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])

data['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].replace(0, np.nan)
data['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].fillna(data['Item_Outlet_Sales'].mode()[0])

# combining reg, Regular and Low Fat, low fat and, LF
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
data['Item_Fat_Content'].value_counts()

# Getting the first two characters of ID to separate them into different categories
data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Identifier'] = data['Item_Identifier'].map({'FD': 'Food', 'NC': 'Non_Consumable', 'DR': 'Drinks'})
data['Item_Identifier'].value_counts()

# determining the operation period of time
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].value_counts()

# Label Encoding
from sklearn.preprocessing import LabelEncoder

data = data.apply(LabelEncoder().fit_transform)

# One-hot Encoding
data = pd.get_dummies(data)

# splitting the data into dependent and independent variables
x = data.drop('Item_Outlet_Sales', axis=1)
y = data.Item_Outlet_Sales

# splitting the dataset into train and test
train = data.iloc[:8523, :]
test = data.iloc[8523:, :]

# making x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Modelling with Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print('RMSE:', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

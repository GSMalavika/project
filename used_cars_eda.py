import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"C:\Users\91940\Downloads\used_cars_data.csv")
print(data)
pd.options.display.max_columns=14
print(data.head())
print(data.tail())
print("\nBasic information about data in each columns\n-------------------")
print(data.info())
print("\nNumber of Unique values in data\n------------------------")
print(data.nunique())
print("\nNumber of nan values in each columns\n-----------------------------")
print(data.isnull().sum())
print("\nPercentage of nan values in each column\n------------------------------")
print((data.isnull().sum()/len(data))*100)

# Removing S NO from data
data=data.drop(['S.No.'],axis=1)
data=data.drop(['New_Price'],axis=1)
print("\nDropping S.No and New_Price columns since S.No has no power over data prediction and New_Price has 86% of nan values\n----------------------------------------")
print(data.info())

print("\nConverting 'Mileage','Engine,'Power' to float type as they are object type due to unit")
def clean_and_convert(column):
    return pd.to_numeric(column.str.extract('(\d+\.?\d*)')[0], errors='coerce')

data['Mileage'] = clean_and_convert(data['Mileage'])
data['Engine'] = clean_and_convert(data['Engine'])
data['Power'] = clean_and_convert(data['Power'])

data['Mileage']=data['Mileage'].astype(float)
data['Engine'] = data['Engine'].astype(float)
data['Power'] = data['Power'].astype(float)

print("\nInfo about Data after converting\n---------------------------")
print(data.info())

#Creating features
print("\nCreating a new column which has age of the car\n------------------------------")
from datetime import date
date.today().year
data['Car_Age']=date.today().year-data['Year']
print(data.head())

print("\n Splitting Name column to Brand and Model\n-------------------------------")
data['Brand']=data.Name.str.split().str.get(0)
data['Model']=data.Name.str.split().str.get(1)+data.Name.str.split().str.get(2)

print(data.Brand)
print(data.Brand.unique())
print(data.Brand.nunique())
print("\n Data replace in brand names")
search_for=['IZUSU','Izusu','Mini','Land']
print(data[data.Brand.str.contains('|'.join(search_for))].head(5))
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper","Land":"Land Rover"}, inplace=True)
print(data)
print("\nData Description which gives statistical analysis of numerical columns\n---------------------------")
print(data.describe().T)

print("\n Data Description of all columns which includes object types also\n--------------------------------")
print(data.describe(include='all').T)

print("Separating Categorical columns and numerical columns\n---------------------------------")
cat_cols=data.select_dtypes(include=['object']).columns.tolist()
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

fig, axes = plt.subplots(3, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset',fontsize=12)
sns.countplot(ax = axes[0, 0], x = 'Fuel_Type', data = data, color = 'skyblue', 
              order = data['Fuel_Type'].value_counts().index);
sns.countplot(ax = axes[0, 1], x = 'Transmission', data = data, color = 'skyblue', 
              order = data['Transmission'].value_counts().index);
sns.countplot(ax = axes[1, 0], x = 'Owner_Type', data = data, color = 'skyblue', 
              order = data['Owner_Type'].value_counts().index);
sns.countplot(ax = axes[1, 1], x = 'Location', data = data, color = 'skyblue', 
              order = data['Location'].value_counts().index);
sns.countplot(ax = axes[2, 0], x = 'Brand', data = data, color = 'skyblue', 
              order = data['Brand'].head(20).value_counts().index);
sns.countplot(ax = axes[2, 1], x = 'Model', data = data, color = 'skyblue', 
              order = data['Model'].head(20).value_counts().index);
axes[1][1].tick_params(labelrotation=45);
axes[2][0].tick_params(labelrotation=90);
axes[2][1].tick_params(labelrotation=90);


for col in num_cols:
    print(col)
    print('Skew :', round(data[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False,color='pink')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col],color='cyan')
    plt.show()

# Function for log transformation of the column
def log_transform(data,col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname]+1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    data.info()
log_transform(data,['Kilometers_Driven','Price'])
#Log transformation of the feature 'Kilometers_Driven'
sns.distplot(data["Kilometers_Driven_log"], axlabel="Kilometers_Driven_log");
sns.distplot(data['Price_log'],axlabel='Price_log');

# Bi-variate analysis of numerical features except Kilometers_Driven and Price
plt.figure(figsize=(13,17))
sns.pairplot(data=data.drop(['Kilometers_Driven','Price'],axis=1),hue='Transmission',hue_order=["Manual","Automatic"])
plt.show()

# Categorical analysis
fig, axarr = plt.subplots(4, 2, figsize=(12, 18))
data.groupby('Location')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)
axarr[0][0].set_title("Location Vs Price", fontsize=18)
data.groupby('Transmission')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
axarr[0][1].set_title("Transmission Vs Price", fontsize=18)
data.groupby('Fuel_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][0], fontsize=12)
axarr[1][0].set_title("Fuel_Type Vs Price", fontsize=18)
data.groupby('Owner_Type')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12)
axarr[1][1].set_title("Owner_Type Vs Price", fontsize=18)
data.groupby('Brand')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][0], fontsize=12)
axarr[2][0].set_title("Brand Vs Price", fontsize=18)
data.groupby('Model')['Price_log'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][1], fontsize=12)
axarr[2][1].set_title("Model Vs Price", fontsize=18)
data.groupby('Seats')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][0], fontsize=12)
axarr[3][0].set_title("Seats Vs Price", fontsize=18)
data.groupby('Car_Age')['Price_log'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][1], fontsize=12)
axarr[3][1].set_title("Car_Age Vs Price", fontsize=18)
plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()

df_num=data._get_numeric_data()
print(df_num)

#EDA multivariate analysis
cmap=sns.diverging_palette(5, 260, as_cmap=True)
plt.figure(figsize=(12, 7))
plt.figure(figsize=(12, 7))
sns.heatmap(df_num.drop(['Kilometers_Driven','Price'],axis=1).corr(),annot=True,cmap=cmap,vmin = -1,vmax = 1)
plt.show()



    
    
    
    
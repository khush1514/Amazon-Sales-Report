# Amazon-Sales-Report
#Analyze and Provide Insights on Amazon Sales Report

#Problem Description:
#The provided dataset contains information about sales transactions on Amazon, including details such as order
#ID, date, status, fulfilment method, sales channel, product category, size, quanƟty, amount, shipping details,
#and more. The objective is to conduct a comprehensive analysis of the data and extract actionable insights to
#support business decision-making.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline  

import pandas as pd

df = pd.read_csv('Amazon Sale Report.csv', encoding = 'latin1')
df.head()

df = df.drop(columns = ['index'])
df.head()

df.isnull().sum()
df

df.duplicated()

num_df = df.select_dtypes(include=['float64','int64'])

#calculate correlation matrix
corr_matrix = num_df.corr()

#display correlation matrix
print(corr_matrix)

df.info()

sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')

df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')

sales_overview = df.groupby('Date').agg({'Amount': 'sum', 'Order ID': 'count'}).reset_index()
sales_overview.rename(columns={'Order ID': 'Number of Orders'}, inplace=True)

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(sales_overview['Date'], sales_overview['Amount'], color='r',marker='o', label='Total Sales')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Sales', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot(sales_overview['Date'], sales_overview['Number of Orders'], color='g',marker='x', label='Number of Orders')
ax2.set_ylabel('Number of Orders', color='g')
ax2.tick_params('y', colors='g')
fig.tight_layout()
plt.title('Total Sales and Number of Orders Over Time')
plt.show()

custom_palette = sns.color_palette("muted",9)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Category', order=df['Category'].value_counts().index,palette=custom_palette)
plt.title('Distribution of Product Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

custom_palette = sns.color_palette("bright",9)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Size', order=df['Size'].value_counts().index,palette=custom_palette)
plt.title('Distribution of Product Sizes')
plt.xlabel('Size')
plt.ylabel('Count')
plt.show()

color_palette = ['#a7c957', '#fb6f92']
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Fulfilment', order=df['Fulfilment'].value_counts().index, palette=color_palette)
plt.title('Distribution of Fulfillment Methods')
plt.xlabel('Fulfillment Method')
plt.ylabel('Count')
plt.show()

custom_palette = sns.color_palette("Paired")
plt.figure(figsize=(6, 4))
fulfillment_effectiveness = df.groupby(['Fulfilment', 'Status']).size().unstack().fillna(0)

fulfillment_effectiveness.plot(kind='bar', stacked=True, figsize=(14, 7),color=custom_palette)
plt.title('Fulfillment Methods Effectiveness')
plt.xlabel('Fulfillment Method')
plt.ylabel('Number of Orders')
plt.legend(title='Order Status')
plt.show()

state_segmentation = df['ship-state'].value_counts().reset_index()
state_segmentation.columns = ['State', 'Number of Orders']
custom_palette = sns.color_palette("hls", len(state_segmentation))
plt.figure(figsize=(14, 7))
sns.barplot(data=state_segmentation, x='State', y='Number of Orders',palette=custom_palette)
plt.title('Customer Segmentation by State')
plt.xlabel('State')
plt.ylabel('Number of Orders')
plt.xticks(rotation=90)
plt.show()

geo_sales = df.groupby(['ship-state', 'ship-city']).agg({'Amount': 'sum'}).reset_index()

state_sales = geo_sales.groupby('ship-state').agg({'Amount': 'sum'}).reset_index()
state_sales = state_sales.sort_values('Amount', ascending=False)
plt.figure(figsize=(14, 7))
sns.barplot(data=state_sales, x='ship-state', y='Amount', palette='colorblind')
plt.title('Sales by State')
plt.xlabel('State')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()

city_sales = geo_sales.groupby('ship-city').agg({'Amount': 'sum'}).reset_index()
city_sales = city_sales.sort_values('Amount', ascending=False).head(10)
plt.figure(figsize=(14, 7))
sns.barplot(data=city_sales, x='ship-city', y='Amount', palette='deep')
plt.title('Sales by City (Top 10)')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()





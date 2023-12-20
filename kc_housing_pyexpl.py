#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# data set

df = pd.read_csv ('kc_housing_data_c.csv')
print (df.info)
print (df.head())


# In[8]:


# calucalting mean, median , min and max sales

# converting to date timeframe
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extracting the year only from the date format
df['year'] = df['date'].dt.year

# filter only for the year 2014
df_2014 = df[df['year'] == 2014]

# Calculate mean, median, min, and max for the 'price' column
mean_price_2014 = df_2014['price'].mean()
median_price_2014 = df_2014['price'].median()
min_price_2014 = df_2014['price'].min()
max_price_2014 = df_2014['price'].max()

# Print the results
print(f"Mean Price (2014): {mean_price_2014}")
print(f"Median Price (2014): {median_price_2014}")
print(f"Min Price (2014): {min_price_2014}")
print(f"Max Price (2014): {max_price_2014}")


# In[33]:


# exploring to see which price range has the most sold properties in year 2014
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# converting to data format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# extracting year from the date
df['year'] = df['date'].dt.year

# data for the year 2014
df_2014 = df[df['year'] == 2014]

# price range defined
price_ranges = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
                550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000, 2000000]

# Create price range categories
df_2014['price_range'] = pd.cut(df_2014['price'], bins=price_ranges, right=False)

# Count the number of houses in each price range
price_range_counts = df_2014['price_range'].value_counts().sort_index().head(10)

# Plotting the top 10 distribution of house prices
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x=price_range_counts.index.astype(str), y=price_range_counts.values, palette='viridis')

# Add labels on top of the bars
for index, value in enumerate(price_range_counts):
    bar_plot.text(index, value + 0.1, str(value), ha='center', va='bottom', fontsize=9)

plt.title('Top 10 Distribution of House Prices in 2014')
plt.xlabel('Price Range')
plt.ylabel('Number of Houses')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# In[34]:


# box plot to see the pricing distribution and observe the outliers

import seaborn as sns
import matplotlib.pyplot as plt

# converting to date format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extracting the year
df['year'] = df['date'].dt.year

# Filter for year 2014
df_2014 = df[df['year'] == 2014]

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='price', data=df_2014, color='skyblue', width=0.5, fliersize=5)

# Adding labels and title
plt.title("Distribution of House Prices in 2014")
plt.xlabel("Price")
plt.ylabel("Price Distribution")
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()


# In[35]:


# analyzing to see if there is any seasonal trend for timing of most sold properties

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Converting to date format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year

df_2014 = df[df['year'] == 2014]

# price ranged defined
price_bins = [0, 200000, 400000, 600000, 800000, np.inf]
price_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df_2014['price_range'] = pd.cut(df_2014['price'], bins=price_bins, labels=price_labels)

# Determine which range had the most sales
most_sales_range = df_2014['price_range'].value_counts().idxmax()

# Set a custom style
plt.style.use('seaborn-whitegrid')

# Visualize the distribution of prices using a histogram
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.hist(df_2014['price'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Prices (2014)')

# Visualize the count of sales in each price range using a bar chart
plt.subplot(1, 2, 2)
bar_colors = plt.cm.viridis(np.arange(len(df_2014['price_range'].value_counts())))
bars = plt.bar(df_2014['price_range'].value_counts().index, df_2014['price_range'].value_counts(), color=bar_colors)
plt.xlabel('Price Range')
plt.ylabel('Number of Sales')
plt.title('Sales Count in Price Range Categories (2014)')

# Adding labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval), ha='center', va='bottom')

plt.tight_layout()  
plt.show()

# result
print(f"The price range with the most sales in 2014 is: {most_sales_range}")


# In[36]:


# 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'bedrooms', 'bathrooms', 'sqft_lot' (size), and 'price' are the column names in your dataset
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year

# Set a custom style
plt.style.use('seaborn-whitegrid')

# Investigate the relationship between house features and sale prices
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x='bedrooms', y='price', data=df, alpha=0.5, edgecolor='w')
plt.title('Bedrooms vs. Sale Price')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Sale Price')
plt.xlim(0, df['bedrooms'].max() + 1)

plt.subplot(1, 3, 2)
sns.scatterplot(x='bathrooms', y='price', data=df, alpha=0.5, edgecolor='w')
plt.title('Bathrooms vs. Sale Price')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Sale Price')
plt.xlim(0, df['bathrooms'].max() + 1)

plt.subplot(1, 3, 3)
sns.scatterplot(x='sqft_lot', y='price', data=df, alpha=0.5, edgecolor='w')  # Change 'size' to 'sqft_lot'
plt.title('Sqft Lot vs. Sale Price')
plt.xlabel('Sqft Lot Size')
plt.ylabel('Sale Price')
plt.xlim(0, df['sqft_lot'].max() + 1000)  # Add some buffer for better visibility

plt.tight_layout()
plt.show()

# Identify the most common features in houses that were sold
most_common_bedrooms = df['bedrooms'].mode().iloc[0]
most_common_bathrooms = df['bathrooms'].mode().iloc[0]
most_common_sqft_lot = df['sqft_lot'].mode().iloc[0]  # Change 'size' to 'sqft_lot'

print(f"The most common features in houses that were sold:")
print(f"Most Common Bedrooms: {most_common_bedrooms}")
print(f"Most Common Bathrooms: {most_common_bathrooms}")
print(f"Most Common Sqft Lot: {most_common_sqft_lot}")  # Change 'size' to 'sqft_lot'


# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'zipcode' and 'price' are the column names in your dataset
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year

# Get the top 10 zip codes by the number of sales
top_zipcodes_by_sales = df['zipcode'].value_counts().head(10).index

# Filter the data for the top 10 zip codes
df_top_zipcodes = df[df['zipcode'].isin(top_zipcodes_by_sales)]

# Analyze the distribution of sales across the top 10 zip codes
plt.figure(figsize=(12, 6))
sales_count_plot = sns.countplot(x='zipcode', data=df_top_zipcodes, order=top_zipcodes_by_sales, palette='viridis')
plt.title('Distribution of Sales Across Top 10 Zip Codes')
plt.xlabel('Zip Code')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Add labels on top of the bars for the count plot
for bar in sales_count_plot.patches:
    sales_count_plot.annotate(f'{int(bar.get_height())}', 
                              (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                              ha='center', va='bottom', fontsize=9, color='black')

plt.show()

# Get the top 10 zip codes by median price
top_zipcodes_by_price = df.groupby('zipcode')['price'].median().sort_values(ascending=False).head(10).index

# Filter the data for the top 10 zip codes by median price
df_top_zipcodes_price = df[df['zipcode'].isin(top_zipcodes_by_price)]

# Determine if there are regions where houses tend to sell for higher prices
plt.figure(figsize=(12, 6))
box_plot = sns.boxplot(x='zipcode', y='price', data=df_top_zipcodes_price, order=top_zipcodes_by_price, palette='viridis')
plt.title('Price Distribution Across Top 10 Zip Codes')
plt.xlabel('Zip Code')
plt.ylabel('Sale Price')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Add labels on top of the boxes for the box plot
for box in box_plot.artists:
    yval = box.get_y() + box.get_height() / 2
    xval = box.get_x() + box.get_width() / 2
    box_plot.annotate(f'{int(yval)}', (xval, yval), ha='center', va='center', fontsize=9, color='black')

plt.show()


# In[21]:


# mapping the distribution

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Assuming 'lat', 'long', and 'price' are the column names in your dataset
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year

# Filter data for the year 2014
df_2014 = df[df['year'] == 2014]

# Create a base map centered around the average coordinates
map_center = [df_2014['lat'].mean(), df_2014['long'].mean()]
mymap = folium.Map(location=map_center, zoom_start=12)

# Create a MarkerCluster to group and display multiple markers efficiently
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers for each house sale
for index, row in df_2014.iterrows():
    folium.Marker([row['lat'], row['long']], popup=f"Price: ${row['price']:,}").add_to(marker_cluster)

# Save the map to an HTML file or display it in a Jupyter notebook
mymap.save("house_sales_map.html")


# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'date' is the name of your date column
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extract the month name from the date
df['month'] = df['date'].dt.strftime('%B')

# Analyze the monthly distribution of house sales using a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='price', data=df, estimator='count', ci=None, marker='o', color='blue')
plt.title('Monthly Distribution of House Sales')
plt.xlabel('Month')
plt.ylabel('Number of Sales')
plt.show()

# Compare the number of sales between 2014 using a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='price', data=df[df['year'] == 2014], estimator='count', ci=None, marker='o', color='green')
plt.title('Monthly Distribution of House Sales in 2014')
plt.xlabel('Month')
plt.ylabel('Number of Sales')
plt.show()


# In[27]:


# seasonal pattern 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'date' is the name of your date column
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extract month and year for additional analysis
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Define seasons based on months
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Create a new 'season' column
df['season'] = df['month'].apply(get_season)

# Filter data for the year 2014
df_2014 = df[df['year'] == 2014]

# Analyze seasonal patterns in house sales for 2014
plt.figure(figsize=(12, 6))
sns.countplot(x='season', data=df_2014, palette='viridis', order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.title('Seasonal Patterns in House Sales (2014)')
plt.xlabel('Season')
plt.ylabel('Number of Sales')
plt.show()

# Identify factors contributing to seasonal variations for 2014
# You can explore additional features, e.g., bedrooms, bathrooms, size
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='price', data=df_2014, palette='viridis', order=['Winter', 'Spring', 'Summer', 'Fall'])
plt.title('Seasonal Variations in House Prices (2014)')
plt.xlabel('Season')
plt.ylabel('Sale Price')
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'size', 'bedrooms', 'bathrooms' are the column names in your dataset
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Extract year for additional analysis
df['year'] = df['date'].dt.year

# Filter data for the year 2014
df_2014 = df[df['year'] == 2014]

# Select relevant columns for correlation analysis
selected_columns = ['sqft_lot', 'bedrooms', 'bathrooms', 'price']  # Adjust as needed
correlation_data = df_2014[selected_columns]

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

# Plot a heatmap to visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Features (2014)')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('kc_housing_data_c.csv')
print(df)


# # Total of all properties sold in the year 2014

# In[9]:


import pandas as pd
import locale

# Set the locale to the default locale of your system
locale.setlocale(locale.LC_ALL, '')

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the year 2014
df_2014 = df.loc[df['date'].dt.year == 2014]

# Sum the 'price' column for the year 2014
total_price_2014 = df_2014['price'].sum()

# Format the total with a dollar sign and comma separation
formatted_total_price_2014 = locale.currency(total_price_2014, grouping=True)

# Print the formatted result
print(f'Total price for the year 2014: {formatted_total_price_2014}')


# # Total of all properties sold in the year 2015
# 

# In[10]:


import pandas as pd
import locale

# Set the locale to the default locale of your system
locale.setlocale(locale.LC_ALL, '')

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the year 2014
df_2015 = df.loc[df['date'].dt.year == 2015]

# Sum the 'price' column for the year 2014
total_price_2015 = df_2015['price'].sum()

# Format the total with a dollar sign and comma separation
formatted_total_price_2015 = locale.currency(total_price_2015, grouping=True)

# Print the formatted result
print(f'Total price for the year 2015: {formatted_total_price_2015}')


# # Total cost of all houses which was sold during year 2014 and 2015
# 

# In[39]:


import pandas as pd
import locale
import matplotlib.pyplot as plt

# Set the locale to the default locale of your system
locale.setlocale(locale.LC_ALL, '')

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the year 2014 and 2015
df_2014 = df.loc[df['date'].dt.year == 2014]
df_2015 = df.loc[df['date'].dt.year == 2015]

# Sum the 'price' column for the years 2014 and 2015
total_price_2014 = df_2014['price'].sum()
total_price_2015 = df_2015['price'].sum()

# Format the totals with a dollar sign and comma separation
formatted_total_price_2014 = locale.currency(total_price_2014, grouping=True)
formatted_total_price_2015 = locale.currency(total_price_2015, grouping=True)

# Print the formatted results
print(f'Total price for the year 2014: {formatted_total_price_2014}')
print(f'Total price for the year 2015: {formatted_total_price_2015}')

# Create a bar graph with dollar values inside the bars
years = ['2014', '2015']
totals = [total_price_2014, total_price_2015]

plt.figure(figsize=(10, 6))

# Bar graph
bars = plt.bar(years, totals, color=['blue', 'green'])

# Add dollar values inside the bars
for bar, total in zip(bars, totals):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{locale.currency(total, grouping=True)}', ha='center', va='bottom')

# Adjustments to make the plot look neater
plt.xlabel('Year')
plt.ylabel('Total Price')
plt.title('Total Cost of Properties for Each Year')
plt.tight_layout()  # Adjust layout for better appearance
plt.show()


# # Number of properties sold at given price range in 2014

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the year 2014
df_2014 = df.loc[df['date'].dt.year == 2014]

# Define price ranges
price_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, float('inf')]
price_labels = ['0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K', '500K-600K', '600K-700K', '700K-800K', '800K-900K', '900K-1M', 'Above 1M']

# Create a new column 'Price Range' based on the defined bins and labels
df_2014['Price Range'] = pd.cut(df_2014['price'], bins=price_bins, labels=price_labels, right=False)

# Count the number of properties in each price range
price_range_counts_2014 = df_2014['Price Range'].value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts_2014.index, y=price_range_counts_2014.values, palette='viridis')
plt.xlabel('Price Range')
plt.ylabel('Number of Properties Sold')
plt.title('Number of Properties Sold in 2014 by Price Range')
plt.xticks(rotation=45, ha='right')  # Adjust x-axis labels for better visibility
plt.show()


# # Number of properties sold at given price range in 2015

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the year 2015
df_2015 = df.loc[df['date'].dt.year == 2015]

# Define price ranges
price_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, float('inf')]
price_labels = ['0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K', '500K-600K', '600K-700K', '700K-800K', '800K-900K', '900K-1M', 'Above 1M']

# Create a new column 'Price Range' based on the defined bins and labels
df_2015['Price Range'] = pd.cut(df_2015['price'], bins=price_bins, labels=price_labels, right=False)

# Count the number of properties in each price range
price_range_counts_2015 = df_2015['Price Range'].value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts_2015.index, y=price_range_counts_2015.values, palette='mako')
plt.xlabel('Price Range')
plt.ylabel('Number of Properties Sold')
plt.title('Number of Properties Sold in 2015 by Price Range')
plt.xticks(rotation=45, ha='right')  # Adjust x-axis labels for better visibility
plt.show()


# # Chart showing the count of sold properties (2014-2015)

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the years 2014 and 2015
df_2014 = df.loc[df['date'].dt.year == 2014]
df_2015 = df.loc[df['date'].dt.year == 2015]

# Define price ranges
price_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, float('inf')]
price_labels = ['0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K', '500K-600K', '600K-700K', '700K-800K', '800K-900K', '900K-1M', 'Above 1M']

# Create new columns 'Price Range' for both years
df_2014['Price Range'] = pd.cut(df_2014['price'], bins=price_bins, labels=price_labels, right=False)
df_2015['Price Range'] = pd.cut(df_2015['price'], bins=price_bins, labels=price_labels, right=False)

# Count the number of properties in each price range for both years
price_range_counts_2014 = df_2014['Price Range'].value_counts().sort_index()
price_range_counts_2015 = df_2015['Price Range'].value_counts().sort_index()

# Get the center positions of the bars
bar_positions = range(len(price_range_counts_2014))
bar_width = 0.7

# Plot a stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(bar_positions, price_range_counts_2014, bar_width, label='2014', color='skyblue')
ax.bar(bar_positions, price_range_counts_2015, bar_width, bottom=price_range_counts_2014, label='2015', color='orange')

ax.set_xlabel('Price Range')
ax.set_ylabel('Count')
ax.set_title('Count of Properties Sold in Different Price Ranges (2014 and 2015)')
ax.legend()

plt.xticks(bar_positions, price_range_counts_2014.index)
plt.xticks(rotation=45)
plt.show()


# # Count of Avg bathroom, bedroom and lot size for the top sold price range of 300k-400k

# In[17]:


import pandas as pd

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the price range 300k-400k
df_price_range = df[(df['price'] >= 300000) & (df['price'] <= 400000)]

# Calculate the average count of bathroom, bedrooms, and sqft lot
average_bathrooms = df_price_range['bathrooms'].mean()
average_bedrooms = df_price_range['bedrooms'].mean()
average_sqft_lot = df_price_range['sqft_lot'].mean()

# Print the results
print(f'Average Bathrooms for Price Range 300k-400k: {average_bathrooms:.2f}')
print(f'Average Bedrooms for Price Range 300k-400k: {average_bedrooms:.2f}')
print(f'Average Sqft Lot for Price Range 300k-400k: {average_sqft_lot:.2f}')


# # Map for the houses ranging from 300k-400k
# 

# In[19]:


pip install folium


# In[23]:


import pandas as pd
import folium

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the price range 300k-400k
df_price_range = df[(df['price'] >= 300000) & (df['price'] <= 400000)]

# Create a map centered around the mean latitude and longitude of the selected houses
mean_lat = df_price_range['lat'].mean()
mean_lon = df_price_range['long'].mean()

# Initialize the map
house_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# Add markers for each house in the price range
for index, row in df_price_range.iterrows():
    folium.Marker([row['lat'], row['long']], popup=f"Price: {row['price']:,}").add_to(house_map)

# Save the map to an HTML file or display it
house_map.save('house_map.html')


# # Correlation Analysis

# In[34]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the price range 300k-400k
df_price_range = df[(df['price'] >= 300000) & (df['price'] <= 400000)]

# Define subsets of numerical features
numerical_features_set1 = ['price', 'sqft_living', 'sqft_lot']
numerical_features_set2 = ['price', 'bedrooms', 'bathrooms']
numerical_features_set3 = ['price', 'condition', 'floors', 'view', 'sqft_basement']

# Calculate correlation matrices for each feature set
correlation_matrix_set1 = df_price_range[numerical_features_set1].corr()
correlation_matrix_set2 = df_price_range[numerical_features_set2].corr()
correlation_matrix_set3 = df_price_range[numerical_features_set3].corr()

# Plot heatmaps to visualize the correlations for each feature set
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(correlation_matrix_set1, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix - Set 1')

plt.subplot(1, 3, 2)
sns.heatmap(correlation_matrix_set2, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix - Set 2')

plt.subplot(1, 3, 3)
sns.heatmap(correlation_matrix_set3, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix - Set 3')

plt.tight_layout()
plt.show()


# # Mean, Median, and Mode

# In[30]:


import pandas as pd
from scipy.stats import mode

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter out prices above 1 million
df_filtered = df[df['price'] <= 1000000]

# Create a new column for price ranges
df_filtered['price_range'] = pd.cut(df_filtered['price'], bins=[0, 300000, 400000, 500000, 1000000],
                                     labels=['<300K', '300K-400K', '400K-500K', '500K-1M'])

# Group by price range and calculate mean, median, and mode
summary_stats_filtered = df_filtered.groupby('price_range')['price'].agg(['mean', 'median', lambda x: mode(x).mode[0]])

# Rename the lambda function column to 'mode'
summary_stats_filtered = summary_stats_filtered.rename(columns={'<lambda_0>': 'mode'})

# Print the results
print(summary_stats_filtered)


# # Looking into the outliers price point

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Box plot to visualize the distribution of house prices within each range
plt.figure(figsize=(12, 8))
sns.boxplot(x='price_range', y='price', data=df)
plt.title('Distribution of House Prices in Different Ranges')
plt.show()


# # Are there any relation between the price and and location of these properties

# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame and parse the 'date' column as datetime
df = pd.read_csv('kc_housing_data_c.csv', parse_dates=['date'])

# Filter rows for the price range from 100k to 1M
df_price_range_all = df[(df['price'] >= 100000) & (df['price'] <= 1000000)]

# Select relevant features for correlation analysis
features_to_compare = ['price', 'zipcode', 'sqft_lot']

# Calculate the correlation matrix
correlation_matrix = df_price_range_all[features_to_compare].corr()

# Plot a heatmap to visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix for Price, Zip Code, and Lot Size')
plt.show()






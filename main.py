# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

# Load and Explore Dataset
data = pd.read_csv('DoorDashData.csv')

# Display initial information

# Display the first five rows
print("First 5 rows of the dataset:")
print(data.head())

# Display the last five rows
print("\nLast 5 rows of the dataset:")
print(data.tail())
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)
print("Missing Values:\n", data.isnull().sum())

# Data Cleaning
# ---------------------
data['created_at'] = pd.to_datetime(data['created_at'])
data['actual_delivery_time'] = pd.to_datetime(data['actual_delivery_time'])
data['created_at'] = data['created_at'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
data['actual_delivery_time'] = data['actual_delivery_time'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
data = data[data['total_items'] != data['total_items'].max()]


monetary_columns = ['subtotal', 'min_item_price', 'max_item_price']
for col in monetary_columns:
    data[col] = data[col] / 100

numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns
data[numerical_columns] = data[numerical_columns].fillna(0)
data = data.dropna(subset=['actual_delivery_time'])
data[categorical_columns] = data[categorical_columns].fillna("Unknown")
outlier_columns = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders']
cleaned_data = data.copy()
for col in outlier_columns:
    cleaned_data = cleaned_data[cleaned_data[col] >= 0]
print("Missing Values after cleaning:\n", cleaned_data.isnull().sum())


# Exploratory Data Analysis (EDA)
# ---------------------------------------
# Summary statistics
eda_summary = cleaned_data.describe()
print("Summary Statistics:\n", eda_summary)

# # Visualize distributions of numerical features
# numerical_columns = ['subtotal', 'total_items', 'num_distinct_items', 'min_item_price', 'max_item_price']
# for column in numerical_columns:
#     plt.figure(figsize=(8, 5))
#     plt.hist(cleaned_data[column], bins=50, edgecolor='k', alpha=0.7)
#     plt.title(f'Distribution of {column}', fontsize=14)
#     plt.xlabel(column, fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.grid(axis='y', alpha=0.75)
#     plt.show()
#
# # Correlation matrix for numerical columns only
# numerical_data = cleaned_data.select_dtypes(include=[np.number])
# correlation_matrix = numerical_data.corr()
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
# plt.title('Correlation Matrix of Numerical Features', fontsize=16)
# plt.show()
#
# # Feature Engineering
# # ---------------------------
# cleaned_data['delivery_time_seconds'] = (
#     cleaned_data['actual_delivery_time'] - cleaned_data['created_at']
# ).dt.total_seconds()
# cleaned_data['order_complexity'] = cleaned_data['total_items'] * cleaned_data['num_distinct_items']
# cleaned_data['dasher_to_order_ratio'] = cleaned_data['total_onshift_dashers'] / (
#     cleaned_data['total_outstanding_orders'] + 1
# )
# cleaned_data['hour_of_day'] = cleaned_data['created_at'].dt.hour
# cleaned_data['day_of_week'] = cleaned_data['created_at'].dt.dayofweek
#
# # Cluster Analysis
# # ------------------------
# clustering_features = ['total_items', 'subtotal', 'order_complexity', 'dasher_to_order_ratio']
# clustering_data = cleaned_data[clustering_features].dropna()
#
# # Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(clustering_data)
#
# # Determine optimal number of clusters using Elbow Method
# inertia = []
# k_values = range(1, 11)
#
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(scaled_data)
#     inertia.append(kmeans.inertia_)
#
# plt.figure(figsize=(8, 5))
# plt.plot(k_values, inertia, marker='o', linestyle='--')
# plt.title('Elbow Method for Optimal Number of Clusters', fontsize=14)
# plt.xlabel('Number of Clusters (k)', fontsize=12)
# plt.ylabel('Inertia', fontsize=12)
# plt.grid(True)
# plt.show()
#
# # Perform clustering with the optimal number of clusters
# optimal_k = 4  # Adjust based on elbow plot
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# clusters = kmeans.fit_predict(scaled_data)
#
# clustering_data['Cluster'] = clusters
#
# # Visualize clusters
# plt.figure(figsize=(10, 7))
# for cluster in np.unique(clusters):
#     cluster_data = clustering_data[clustering_data['Cluster'] == cluster]
#     plt.scatter(cluster_data['total_items'], cluster_data['subtotal'], label=f'Cluster {cluster}')
#
# plt.title('Cluster Analysis: Total Items vs Subtotal', fontsize=14)
# plt.xlabel('Total Items', fontsize=12)
# plt.ylabel('Subtotal (scaled)', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 7))
# for cluster in np.unique(clusters):
#     cluster_data = clustering_data[clustering_data['Cluster'] == cluster]
#     plt.scatter(cluster_data['order_complexity'], cluster_data['dasher_to_order_ratio'], label=f'Cluster {cluster}')
#
# plt.title('Cluster Analysis: Order Complexity vs Dasher-to-Order Ratio', fontsize=14)
# plt.xlabel('Order Complexity', fontsize=12)
# plt.ylabel('Dasher-to-Order Ratio (scaled)', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.scatter(
#     clustering_data[clustering_data['Cluster'] == 0]['total_items'],
#     clustering_data[clustering_data['Cluster'] == 0]['subtotal'],
#     label='Cluster 0',
#     color='blue'
# )
# plt.xlabel('Total Items')
# plt.ylabel('Subtotal')
# plt.title('Cluster 0: Total Items vs Subtotal')
# plt.legend()
# plt.show()
#
# #Association Analysis to find frequent itemsets
# # We have sliced the dataset and used Apriori algorithm with chunk processing due to memory limits
#
# # Halving the dataset
# halved_data = cleaned_data.sample(frac=0.5, random_state=42)
#
# # Add num_itemsets to association_rules() if required
# try:
#     # Process popular stores as transactions
#     #Filters out stores with 10 or fewer transactions, retaining only the “popular” stores with more than 10 transactions.
#     popular_stores = halved_data['store_id'].value_counts()
#     popular_stores = popular_stores[popular_stores > 10].index
#     filtered_data = halved_data[halved_data['store_id'].isin(popular_stores)]
#
#     chunk_size = 10  # Define manageable chunk size
#     num_chunks = (len(filtered_data) // chunk_size) + 1
#
#     combined_frequent_itemsets = pd.DataFrame()
#     combined_rules = pd.DataFrame()
#
#     for i in range(num_chunks):
#         print(f"Processing chunk {i + 1}/{num_chunks}...")
#
#         chunk = filtered_data.iloc[i * chunk_size:(i + 1) * chunk_size]
#
#         transaction_data = chunk.groupby(['store_id', 'num_distinct_items'])['total_items'].sum().unstack().fillna(0)
#         transaction_data = transaction_data.applymap(lambda x: 1 if x > 0 else 0)
#
#         frequent_itemsets = apriori(transaction_data, min_support=0.05, use_colnames=True, low_memory=True)
#         frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
#
#         if frequent_itemsets.empty:
#             print(f"No frequent itemsets in chunk {i + 1}.")
#             continue
#
#         combined_frequent_itemsets = pd.concat([combined_frequent_itemsets, frequent_itemsets], axis=0)
#
#         # Add num_itemsets argument if required
#         frequent_itemsets['num_itemsets'] = frequent_itemsets['itemsets'].apply(len)
#         rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2, num_itemsets=len(frequent_itemsets))
#
#         combined_rules = pd.concat([combined_rules, rules], axis=0)
#
#     combined_frequent_itemsets = combined_frequent_itemsets.drop_duplicates().reset_index(drop=True)
#     combined_rules = combined_rules.drop_duplicates().reset_index(drop=True)
#
#     # Visualize combined rules
#     if not combined_rules.empty:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(combined_rules['support'], combined_rules['confidence'], alpha=0.6, edgecolor='k', c=combined_rules['lift'], cmap='viridis')
#         plt.colorbar(label='Lift')
#         plt.title('Association Rules: Support vs Confidence', fontsize=14)
#         plt.xlabel('Support', fontsize=12)
#         plt.ylabel('Confidence', fontsize=12)
#         plt.grid(alpha=0.5)
#         plt.show()
#     else:
#         print("No valid association rules generated.")
#
#     # Output results
#     print("Top 10 Combined Association Rules:\n", combined_rules.head(10))
#     print("Top 10 Combined Frequent Itemsets:\n", combined_frequent_itemsets.head(10))
#
# except Exception as e:
#     print("An error occurred during Association Analysis:", str(e))
#
# #Data Visualization (Pending types):
#
# # Column Plot: Frequency of Transactions by Store Category
# store_category_counts = cleaned_data['store_primary_category'].value_counts()
#
# plt.figure(figsize=(10, 6))
# store_category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('Frequency of Transactions by Store Category', fontsize=14)
# plt.xlabel('Store Primary Category', fontsize=12)
# plt.ylabel('Number of Transactions', fontsize=12)
# plt.xticks(rotation=45)
# plt.grid(axis='y', alpha=0.75)
# plt.show()
#
# # Pie Plot: Proportion of Distinct Items Sold by Top Stores
# top_stores = cleaned_data['store_id'].value_counts().head(5).index
# top_store_data = cleaned_data[cleaned_data['store_id'].isin(top_stores)]
# distinct_item_counts = top_store_data.groupby('store_id')['num_distinct_items'].sum()
#
# plt.figure(figsize=(8, 8))
# distinct_item_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, cmap='viridis')
# plt.title('Proportion of Distinct Items Sold by Top Stores', fontsize=14)
# plt.ylabel('')  # Remove y-axis label for pie plot
# plt.show()
#
# # Line Plot: Total Items Sold by Hour of the Day
# hourly_sales = cleaned_data.groupby('hour_of_day')['total_items'].sum()
#
# plt.figure(figsize=(10, 6))
# hourly_sales.plot(kind='line', marker='o', color='green')
# plt.title('Total Items Sold by Hour of the Day', fontsize=14)
# plt.xlabel('Hour of the Day', fontsize=12)
# plt.ylabel('Total Items Sold', fontsize=12)
# plt.grid(alpha=0.5)
# plt.show()
#
#
# # Multi-Line Plot: Busy Dashers vs Outstanding Orders by Hour of the Day
# hourly_data = cleaned_data.groupby('hour_of_day').agg({
#     'total_busy_dashers': 'sum',
#     'total_outstanding_orders': 'sum'
# })
#
# plt.figure(figsize=(10, 6))
# plt.plot(hourly_data.index, hourly_data['total_busy_dashers'], label='Total Busy Dashers', marker='o')
# plt.plot(hourly_data.index, hourly_data['total_outstanding_orders'], label='Total Outstanding Orders', marker='o')
# plt.title('Busy Dashers vs Outstanding Orders by Hour of the Day', fontsize=14)
# plt.xlabel('Hour of the Day', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.legend()
# plt.grid(alpha=0.5)
# plt.show()
#
# # Boxplot: Distribution of Total Items Ordered by Store Category
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=cleaned_data, x='store_primary_category', y='total_items', showfliers=False)
# plt.title('Distribution of Total Items Ordered by Store Category', fontsize=14)
# plt.xlabel('Store Primary Category', fontsize=12)
# plt.ylabel('Total Items Ordered', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.5)
# plt.tight_layout()
# plt.show()
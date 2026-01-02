import pandas as pd
import numpy as np
from scipy import stats
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
df = pd.read_csv('DoorDashData.csv')
print(df.isnull().sum())

df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

df['delivery_time_minutes'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60

df = df.drop(columns=['market_id', 'created_at', 'actual_delivery_time', 'store_id'], axis=1)

df.head(2)

df.nunique()

df['delivery_time_minutes'].describe().T

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print(df['delivery_time_minutes'].isnull().sum())


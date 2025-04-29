# data_wrangling_titanic.py

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 2. Load the Dataset
# Make sure the 'titanic.csv' file is in the same directory as this script
df = pd.read_csv('titanic.csv')
print("Dataset Loaded Successfully!")

# 3. Initial Data Exploration
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe(include='all'))

# 4. Variable Descriptions and Data Types
print("\n--- Variable Descriptions ---")
for column in df.columns:
    print(f"{column}: {df[column].dtype}")

# 5. Data Formatting and Type Conversion
print("\n--- Data Types Before ---")
print(df.dtypes)

# Convert appropriate columns to categorical
df['Pclass'] = df['Pclass'].astype('category')
df['Survived'] = df['Survived'].astype('category')

# Convert object columns to categorical
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

print("\n--- Data Types After ---")
print(df.dtypes)

# 6. Categorical to Quantitative Conversion
# Example using LabelEncoder
le = LabelEncoder()
if 'Sex' in df.columns:
    df['Sex_encoded'] = le.fit_transform(df['Sex'])
if 'Embarked' in df.columns:
    df['Embarked_encoded'] = le.fit_transform(df['Embarked'].astype(str))

# OR using pd.get_dummies
df_dummies = pd.get_dummies(df, drop_first=True)

print("\n--- Encoded Data Sample ---")
print(df_dummies.head())

# Optional: Visualize missing values
print("\n--- Missing Values Heatmap ---")
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

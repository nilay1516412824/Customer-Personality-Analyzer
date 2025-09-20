# Customer Personality Analyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
file_path = r'C:\nilay code\Projects\cpa\marketing_campaign.csv'
data = pd.read_csv(file_path, sep="\t")   # dataset uses tab separator

print("First 5 rows of dataset:")
print(data.head())
print("\nColumns:", data.columns)

# 2. Basic cleaning
data = data.dropna()   # drop missing values
data = data.drop_duplicates()

# Create Age feature
data['Age'] = 2025 - data['Year_Birth']

# Total spending
data['TotalSpend'] = data[['MntWines','MntFruits','MntMeatProducts',
                           'MntFishProducts','MntSweetProducts',
                           'MntGoldProds']].sum(axis=1)

# 3. Clustering (Customer Segmentation)
features = data[['Age','Income','TotalSpend']]
features = features.dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Find best K using silhouette score
scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    scores[k] = silhouette_score(scaled_features, kmeans.labels_)

best_k = max(scores, key=scores.get)
print("Best K (clusters):", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# 4. Churn Prediction
# Assume churn if no purchases in last 2 years
data['Churn'] = np.where(data['Recency'] > 60, 1, 0)

X = data[['Age','Income','TotalSpend']]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Churn Prediction Accuracy:", model.score(X_test, y_test))

# 5. Personas
persona_summary = data.groupby('Cluster').agg({
    'Age':'mean',
    'Income':'mean',
    'TotalSpend':'mean',
    'Churn':'mean'
}).reset_index()

print("\nCustomer Personas:")
print(persona_summary)

# 6. Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Income', y='TotalSpend', hue='Cluster', palette='Set2')
plt.title("Customer Segments")
plt.show()

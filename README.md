# Customer-Personality-Analyzer
Build an AI system that automatically groups customers and predicts their behavior

**Used Dataset Attached**

Live-Demo Link: https://replit.com/@nilay1516412824/Customer-Personality-Analyzer

Demo Video: https://www.loom.com/share/55d2a101f73042faa692513976c394c8?sid=8e35c918-d75c-4b79-9cec-0fa780ae05d7





**Setup Guide**

1.Install ML Libraries

• scikit-learn

• matplotlib

• seaborn

2.Create ML Accounts

• Weights & Biases(track your AI experiments)

• Streamlit Cloud(deploy your apps)

**Core Features:**

~ Analyzes customer shopping patterns

~ Automatically finds different customer types (clusters)

~ Predicts which customers might stop buying (churn prediction)

~ Creates customer personas with descriptions

~ Recommends marketing strategies for each group

~ Dashboard showing insights and recommendations


**Dataset**

The project utilizes the marketing_campaign.csv dataset, which contains customer information such as:

**Demographic Data:** Age, education, marital status, income

**Behavioral Data:** Spending habits, purchase history

**Customer Activity:** Recency, frequency, and monetary (RFM) metrics

**Models Used**

The system employs several machine learning models to analyze and predict customer behavior:

**1.K-Means Clustering:** Used for customer segmentation based on purchasing behavior and demographics.

**2.Random Forest Classifier:** Applied to predict customer churn and identify high-value customers.

**3.Logistic Regression:** Utilized for binary classification tasks, such as predicting the likelihood of a customer making a purchase.

**App Workflow**

**1.Data Preprocessing:** The dataset is cleaned and preprocessed to handle missing values, encode categorical variables, and normalize numerical features.

**2.Feature Engineering:** Relevant features are selected and transformed to enhance model performance.

**3.Model Training:** The models are trained on the preprocessed data, with hyperparameters tuned for optimal accuracy.

**4.Prediction & Segmentation:** The trained models predict customer behavior, and K-Means clustering segments customers into distinct groups.

**5.Visualization:** Results are visualized using matplotlib and seaborn for easy interpretation.







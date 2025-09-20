import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="Customer Personality Analyzer", layout="wide")

@st.cache_data
def load_data(path=r'C:\nilay code\Projects\cpa\marketing_campaign.csv'):
    # try common separators
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding='utf-8')
            if df.shape[1] > 3:
                return df
        except Exception:
            continue
    return pd.read_csv(path)

@st.cache_data
def preprocess(df):
    df = df.copy()
    df = df.loc[:, df.notna().sum() > 0]
    df.drop_duplicates(inplace=True)

    if 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']

    mnt_cols = [c for c in df.columns if c.startswith('Mnt')]
    if len(mnt_cols) > 0:
        df['TotalSpend'] = df[mnt_cols].sum(axis=1)
        for c in mnt_cols:
            df[f'{c}_pct'] = df[c] / (df['TotalSpend'].replace(0, np.nan))
            df[f'{c}_pct'].fillna(0, inplace=True)
    else:
        df['TotalSpend'] = 0

    if 'Income' in df.columns:
        df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
        df['Income'].fillna(df['Income'].median(), inplace=True)
    else:
        df['Income'] = df['TotalSpend']

    if 'Recency' not in df.columns:
        df['Recency'] = 30

    if 'NumDealsPurchases' in df.columns:
        df['NumPurchases'] = df.get('NumWebPurchases', 0) + df.get('NumStorePurchases', 0) + df.get('NumCatalogPurchases', 0)
    else:
        df['NumPurchases'] = 0

    df['Churn'] = 0
    if 'Recency' in df.columns:
        df['Churn'] = (df['Recency'] >= 90).astype(int)

    return df

@st.cache_data
def compute_segmentation(df, features, k_min=2, k_max=6):
    X = df[features].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = 3
    best_score = -1
    scores = {}
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

    labels = best_kmeans.predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(Xs)
    return labels, components, best_k, scores

@st.cache_data
def train_churn_model(df, features, target='Churn'):
    df2 = df.copy()
    X = df2[features].fillna(df2[features].median())
    y = df2[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique()>1 else None)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, report, cm

# -----------------
# App layout
# -----------------
st.title("🧠 Customer Personality Analyzer")

# Load data
with st.spinner("Loading dataset..."):
    try:
        df_raw = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

st.sidebar.header("Controls")
show_raw = st.sidebar.checkbox("Show raw data (first 50 rows)")
if show_raw:
    st.dataframe(df_raw.head(50))

# Preprocess
df = preprocess(df_raw)

st.sidebar.markdown("---")
default_features = ['Age','Income','TotalSpend']
available_feats = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] or c in default_features]
selected_feats = st.sidebar.multiselect("Features for segmentation (numeric)", options=available_feats, default=default_features)

st.sidebar.markdown("---")
recency_thresh = st.sidebar.slider("Churn Recency Threshold (days)", min_value=30, max_value=365, value=90)
df['Churn'] = (df['Recency'] >= recency_thresh).astype(int)

# Segmentation
if len(selected_feats) < 2:
    st.warning("Please select at least 2 numeric features for clustering.")
else:
    labels, components, best_k, scores = compute_segmentation(df, selected_feats)
    df['Segment'] = labels

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Customer Segments (PCA projection)")
        fig = px.scatter(
            x=components[:, 0], 
            y=components[:, 1], 
            color=df['Segment'].astype(str), 
            hover_data={
                'Index': df.index,
                'Age': df['Age'],
                'Income': df['Income'],
                'TotalSpend': df['TotalSpend']
            }
        )
        fig.update_layout(xaxis_title='PC1', yaxis_title='PC2')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster selection & scores")
        st.write(f"Suggested optimum clusters (by silhouette): {best_k}")
        st.write("Silhouette scores by k:")
        st.table(pd.Series(scores).rename('silhouette').to_frame())

    # Cluster profiles
    st.subheader("Cluster profiles / Personas")
    personas = df.groupby('Segment').agg({
        'Age': 'mean',
        'Income': 'mean',
        'TotalSpend': 'mean',
        'Churn': 'mean',
        'NumPurchases': 'mean'
    }).round(2).reset_index()
    personas.rename(columns={
        'Churn': 'ChurnRate',
        'NumPurchases': 'AvgNumPurchases'
    }, inplace=True)
    st.dataframe(personas)

    # Persona Descriptions & Recommendations
    st.subheader("Persona descriptions & marketing recommendations")
    persona_texts = []
    for _, row in personas.iterrows():
        seg = int(row['Segment'])
        desc = f"Segment {seg}: Avg Age {row['Age']:.0f}, Avg Income {row['Income']:.0f}, Avg Spend {row['TotalSpend']:.0f}, ChurnRate {row['ChurnRate']:.2f}"
        
        # Recommendations
        recs = []
        if row['Income'] > personas['Income'].median() and row['TotalSpend'] < personas['TotalSpend'].median():
            recs.append("High-income but low spend — suggest upsell/premium bundles.")
        if row['TotalSpend'] > personas['TotalSpend'].median():
            recs.append("High spenders — focus on loyalty and retention (VIP offers).")
        if row['ChurnRate'] > 0.2:
            recs.append("High churn risk — re-engagement emails, time-limited discounts.")
        if not recs:
            recs.append("No special recommendation — standard engagement.")

        full_text = f"{desc}\nRecommendations: {'; '.join(recs)}"
        persona_texts.append(full_text)

    for text in persona_texts:
        st.markdown(f"• {text}")

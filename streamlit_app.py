
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("Retail Data Clustering")

# File uploader for the Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_excel(uploaded_file, sheet_name=0)
    st.write("Data Sample")
    st.dataframe(df.head())
    
    # Preprocess data (drop missing values, etc.)
    df_clean = df.dropna()
    
    # Feature selection (customize according to the dataset)
    feature_columns = ['Quantity', 'Price']
    
    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean[feature_columns])
    
    # Clustering options
    st.sidebar.header("Clustering Options")
    clustering_type = st.sidebar.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN"])
    
    if clustering_type == "KMeans":
        # Number of clusters
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        df_clean['Cluster'] = kmeans.fit_predict(scaled_data)
    elif clustering_type == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
        dbscan = DBSCAN(eps=eps)
        df_clean['Cluster'] = dbscan.fit_predict(scaled_data)
    
    # Visualization
    st.subheader(f"{clustering_type} Clustering Results")
    fig = px.scatter(df_clean, x='Quantity', y='Price', color='Cluster', title=f"{clustering_type} Clusters")
    st.plotly_chart(fig)

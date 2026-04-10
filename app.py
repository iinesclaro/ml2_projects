import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Spotify Clustering - Elbow Method", layout="wide")

st.title("🎵 Spotify Clustering - Elbow Method")
st.markdown("Interactive visualization of the elbow method for optimal K-means cluster selection")

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Load dataset
    df_spotify = pd.read_csv("dataset.csv")
    df_spotify.drop("Unnamed: 0", axis=1, inplace=True)
    df_spotify.drop(df_spotify[df_spotify['artists'].isna()].index, axis=0, inplace=True)
    df_spotify.reset_index(drop=True, inplace=True)
    
    # Select features for clustering
    spotify_clustering = df_spotify[[
        'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'time_signature'
    ]].copy()
    
    # Log transformations
    spotify_clustering['log_duration_ms'] = np.log(spotify_clustering['duration_ms'])
    spotify_clustering['log_speechiness'] = np.log1p(spotify_clustering['speechiness'])
    
    # Drop mode
    spotify_clustering.drop(columns=['mode'], inplace=True)
    
    # Scale the data
    scaler = StandardScaler()
    spotify_clustering_scaled = scaler.fit_transform(spotify_clustering)
    
    return spotify_clustering_scaled

# Load data
spotify_clustering_scaled = load_and_prepare_data()

# Sidebar for user input
st.sidebar.header("Settings")
max_clusters = st.sidebar.slider("Maximum number of clusters to test:", 5, 30, 20)

# Calculate inertias
@st.cache_data
def calculate_inertias(max_clusters):
    inertias = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
        kmeans.fit(spotify_clustering_scaled)
        inertias.append(kmeans.inertia_)
    return inertias

inertias = calculate_inertias(max_clusters)

# Create columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Elbow Method Curve")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_clusters + 1), inertias, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
    ax.fill_between(range(1, max_clusters + 1), inertias, alpha=0.2, color='#1f77b4')
    ax.set_title('Elbow Method For Optimal K', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_clusters + 1, 2))
    st.pyplot(fig)

with col2:
    st.subheader("Top Clusters")
    # Calculate differences to find elbow
    differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    second_differences = [differences[i] - differences[i+1] for i in range(len(differences)-1)]
    
    top_k = sorted(range(len(second_differences)), key=lambda i: second_differences[i], reverse=True)[:3]
    
    for idx, k in enumerate(sorted(top_k), 1):
        st.metric(f"Candidate {idx}", f"K = {k+2}", f"Inertia: {inertias[k+1]:.2f}")

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Inertia Table")
    inertia_df = pd.DataFrame({
        'K (Clusters)': range(1, max_clusters + 1),
        'Inertia': [f'{x:.2f}' for x in inertias]
    })
    st.dataframe(inertia_df, use_container_width=True, hide_index=True)

with col4:
    st.subheader("About the Elbow Method")
    st.info("""
    **What is the Elbow Method?**
    
    It helps determine the optimal number of clusters by finding the point where adding more clusters doesn't significantly reduce inertia.
    
    **Key Points:**
    - Look for the "elbow" or "knee" in the curve
    - Beyond this point, inertia decreases slowly
    - For this Spotify dataset, the elbow typically appears around K=8-10
    - The candidates shown represent possible optimal values
    """)

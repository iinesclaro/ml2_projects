import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Spotify Clustering - Elbow Method", layout="wide")

st.title("🎵 Spotify Clustering - Elbow Method")
st.markdown("Interactive visualization of the elbow method for optimal K-means cluster selection")

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
    
    return spotify_clustering_scaled, spotify_clustering, scaler

class SelfOrganizingMap:
    def __init__(self, m, n, dim, learning_rate=0.5, sigma=None, random_seed=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(m, n) / 2.0
        self.random_state = np.random.RandomState(random_seed)
        self.weights = self.random_state.rand(m * n, dim)
        self.locations = np.array([[i, j] for i in range(m) for j in range(n)])

    def _decay(self, initial_value, iteration, max_iter):
        return initial_value * np.exp(-iteration / max_iter)

    def winner(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)

    def train(self, data, num_iterations=100):
        for iteration in range(num_iterations):
            lr = self._decay(self.learning_rate, iteration, num_iterations)
            sigma = self._decay(self.sigma, iteration, num_iterations)
            for sample in data:
                winner_idx = self.winner(sample)
                dist_to_winner = np.linalg.norm(self.locations - self.locations[winner_idx], axis=1)
                influence = np.exp(-(dist_to_winner ** 2) / (2 * (sigma ** 2)))
                self.weights += lr * influence[:, np.newaxis] * (sample - self.weights)

    def map_vects(self, data):
        mapped = [self.locations[self.winner(sample)] for sample in data]
        return np.array(mapped)

    def u_matrix(self):
        umatrix = np.zeros(self.m * self.n)
        for idx, loc in enumerate(self.locations):
            neighbor_dists = []
            for jdx, nloc in enumerate(self.locations):
                if np.sum(np.abs(loc - nloc)) == 1:
                    neighbor_dists.append(np.linalg.norm(self.weights[idx] - self.weights[jdx]))
            umatrix[idx] = np.mean(neighbor_dists) if neighbor_dists else 0
        return umatrix.reshape(self.m, self.n)

# Load data
spotify_clustering_scaled, spotify_clustering, scaler = load_and_prepare_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Elbow Method", "Cluster Analysis", "SOM Analysis"])

with tab1:
    # Sidebar for user input
    st.sidebar.header("Settings")
    max_clusters = st.sidebar.slider("Maximum number of clusters to test:", 5, 20, 12)

    # Calculate inertias and silhouette scores
    @st.cache_data
    def calculate_metrics(max_clusters):
        inertias = []
        silhouettes = []
        sample_size = min(5000, spotify_clustering_scaled.shape[0])

        for i in range(1, max_clusters + 1):
            kmeans = MiniBatchKMeans(
                n_clusters=i,
                random_state=0,
                batch_size=1024,
                max_iter=100,
                init='k-means++',
                n_init=3
            )
            labels = kmeans.fit_predict(spotify_clustering_scaled)
            inertias.append(kmeans.inertia_)
            if i > 1:
                if spotify_clustering_scaled.shape[0] > sample_size:
                    silhouettes.append(
                        silhouette_score(
                            spotify_clustering_scaled,
                            labels,
                            sample_size=sample_size,
                            random_state=0
                        )
                    )
                else:
                    silhouettes.append(silhouette_score(spotify_clustering_scaled, labels))
            else:
                silhouettes.append(np.nan)
        return inertias, silhouettes

    with st.spinner('Computing K-means metrics...'):
        inertias, silhouettes = calculate_metrics(max_clusters)

    # Create columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Elbow + Silhouette Analysis")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        x = np.arange(1, max_clusters + 1)
        ax1.plot(x, inertias, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Inertia')
        ax1.fill_between(x, inertias, alpha=0.15, color='#1f77b4')
        ax2.plot(x[1:], silhouettes[1:], marker='s', linewidth=2.5, markersize=7, color='#ff7f0e', label='Silhouette')

        ax1.set_title('Elbow Method and Silhouette Coefficient for K-means', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12, color='#1f77b4')
        ax2.set_ylabel('Silhouette Coefficient', fontsize=12, color='#ff7f0e')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, max_clusters + 1, 2))
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        st.pyplot(fig)

    with col2:
        st.subheader("Candidate K Values")
        differences = [inertias[i] - inertias[i+1] for i in range(len(inertias) - 1)]
        inertia_drop_norm = np.array(differences) / max(differences) if max(differences) > 0 else np.zeros_like(differences)
        silhouette_norm = np.array([(s - np.nanmin(silhouettes[1:])) / (np.nanmax(silhouettes[1:]) - np.nanmin(silhouettes[1:])) if not np.isnan(s) else 0 for s in silhouettes[1:]])
        combined_scores = 0.55 * silhouette_norm + 0.45 * inertia_drop_norm

        candidate_indices = np.argsort(combined_scores)[::-1][:3] + 2
        for idx, k in enumerate(sorted(candidate_indices), 1):
            st.metric(
                f"Trade-off Candidate {idx}",
                f"K = {k}",
                f"Silhouette: {silhouettes[k-1]:.3f} · Inertia: {inertias[k-1]:.2f}"
            )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("K-means Metrics Table")
        metrics_df = pd.DataFrame({
            'K (Clusters)': range(1, max_clusters + 1),
            'Inertia': [f'{x:.2f}' for x in inertias],
            'Silhouette': [f'{x:.3f}' if not np.isnan(x) else '-' for x in silhouettes]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        if spotify_clustering_scaled.shape[0] > 5000:
            st.caption("Silhouette scores are estimated on a sample of up to 5,000 rows, and MiniBatchKMeans is used for faster elbow metric computation.")

    with col4:
        st.subheader("About the Elbow Method")
        st.info("""
        **What is the Elbow Method?**
        
        It helps determine the optimal number of clusters by finding the point where adding more clusters doesn't significantly reduce inertia.
        
        **Key Points:**
        - Look for the "elbow" or "knee" in the curve
        - Beyond this point, inertia decreases slowly
        - A good trade-off uses both inertia reduction and silhouette score
        - The candidates shown represent balanced values between compact clusters and cluster separation
        """)

with tab2:
    st.header("Cluster Analysis")
    st.markdown("Choose the number of clusters (k) and analyze the resulting clusters.")
    
    # User input for k
    k = st.slider("Select number of clusters (k):", min_value=2, max_value=20, value=8, step=1)
    
    # Perform K-means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(spotify_clustering_scaled)
    
    # Add cluster labels to original data
    spotify_with_clusters = spotify_clustering.copy()
    spotify_with_clusters['cluster'] = clusters
    
    # Cluster sizes
    cluster_sizes = spotify_with_clusters['cluster'].value_counts().sort_index()
    
    st.subheader(f"Cluster Summary for k={k}")
    
    # Display cluster sizes
    st.write("**Number of observations in each cluster:**")
    sizes_df = pd.DataFrame({
        'Cluster': cluster_sizes.index,
        'Number of Observations': cluster_sizes.values
    })
    st.dataframe(sizes_df, use_container_width=True, hide_index=True)
    
    # Centroids in original scale
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    centroids_df = pd.DataFrame(centroids_original, columns=spotify_clustering.columns)
    centroids_df.index.name = 'Cluster'
    centroids_df.reset_index(inplace=True)
    
    st.write("**Cluster Means (Centroids) in Original Scale:**")
    st.dataframe(centroids_df, use_container_width=True, hide_index=True)
    
    # Optional: Visualize clusters (e.g., PCA for 2D plot)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(spotify_clustering_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['cluster'] = clusters
    
    st.subheader("Cluster Visualization (PCA)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in range(k):
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', alpha=0.6)
    ax.set_title(f'Clusters Visualization (k={k})')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.header("Self-Organizing Map (SOM)")
    st.markdown("Use a SOM grid to organize the Spotify tracks by similarity and visualize neuron assignments.")

    som_rows = st.slider("SOM grid rows:", min_value=5, max_value=20, value=10, step=1)
    som_cols = st.slider("SOM grid columns:", min_value=5, max_value=20, value=10, step=1)
    num_iterations = st.slider("Training iterations:", min_value=10, max_value=300, value=120, step=10)
    train_som = st.button("Train SOM")

    if train_som:
        som = SelfOrganizingMap(
            som_rows,
            som_cols,
            spotify_clustering_scaled.shape[1],
            learning_rate=0.5,
            sigma=max(som_rows, som_cols) / 2.0,
            random_seed=0
        )
        som.train(spotify_clustering_scaled, num_iterations=num_iterations)
        mapped = som.map_vects(spotify_clustering_scaled)
        som_labels = mapped[:, 0] * som_cols + mapped[:, 1]

        st.subheader("SOM Neuron Assignments")
        assignment_counts = pd.Series(som_labels).value_counts().sort_index()
        assignment_df = pd.DataFrame({
            'Neuron ID': assignment_counts.index,
            'Mapped Samples': assignment_counts.values
        })
        st.dataframe(assignment_df.head(12), use_container_width=True, hide_index=True)

        most_populated = assignment_counts.nlargest(5)
        st.write("**Top 5 Most Populated Neurons:**")
        st.write(most_populated)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(spotify_clustering_scaled)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        pca_df['neuron'] = som_labels

        st.subheader("SOM Projection (PCA)")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            pca_df['PC1'],
            pca_df['PC2'],
            c=pca_df['neuron'],
            cmap='tab20',
            alpha=0.7,
            s=30
        )
        ax.set_title('SOM Node Assignments in PCA Space')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SOM neuron id')
        st.pyplot(fig)

        st.subheader("SOM U-Matrix")
        umatrix = som.u_matrix()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(umatrix, cmap='viridis', origin='lower')
        ax2.set_title('SOM U-Matrix')
        ax2.set_xlabel('Neuron column')
        ax2.set_ylabel('Neuron row')
        fig2.colorbar(im, ax=ax2, label='Average neighbor distance')
        st.pyplot(fig2)

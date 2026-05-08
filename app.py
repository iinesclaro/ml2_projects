import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
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
    
    # Select features for clustering (matching notebook)
    features = ['popularity', 'danceability', 'energy',
                'liveness', 'valence', 'tempo']
    
    spotify_clustering = df_spotify[features].copy()
    
    # Log transformations (matching notebook)
    spotify_clustering['log_speechiness'] = np.log1p(df_spotify['speechiness'])
    spotify_clustering['log_instrumentalness'] = np.log1p(df_spotify['instrumentalness'])
    spotify_clustering['log_duration_ms'] = np.log(df_spotify['duration_ms'])
    
    # Scale the data using RobustScaler (matching notebook)
    scaler = RobustScaler()
    spotify_clustering_scaled = scaler.fit_transform(spotify_clustering)
    
    return spotify_clustering_scaled, spotify_clustering, scaler

@st.cache_data
def compute_assignment_counts_cached(som_labels):
    assignment_counts = pd.Series(som_labels).value_counts().sort_index()
    return assignment_counts

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

@st.cache_data
def train_som_cached(som_rows, som_cols, num_iterations, data):
    som = SelfOrganizingMap(
        som_rows,
        som_cols,
        data.shape[1],
        learning_rate=0.5,
        sigma=max(som_rows, som_cols) / 2.0,
        random_seed=0
    )
    som.train(data, num_iterations=num_iterations)
    mapped = som.map_vects(data)
    som_labels = mapped[:, 0] * som_cols + mapped[:, 1]
    return som, mapped, som_labels

@st.cache_data
def compute_u_matrix_cached(som):
    return som.u_matrix()

@st.cache_data
def compute_population_grid_cached(som_labels, som_rows, som_cols):
    som_map = np.zeros((som_rows, som_cols))
    for idx, label in enumerate(som_labels):
        row = int(label // som_cols)
        col = int(label % som_cols)
        som_map[row, col] += 1
    return som_map

# Load data
spotify_clustering_scaled, spotify_clustering, scaler = load_and_prepare_data()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Elbow Method", "K=6 Metrics & Radar", "Cluster Analysis", "Song Search", "SOM Analysis"])

with tab1:
    # Sidebar for user input
    st.sidebar.header("Settings")
    max_clusters = st.sidebar.slider("Maximum number of clusters to test:", 5, 20, 12)

    # Calculate inertias and silhouette scores
    @st.cache_data
    def calculate_metrics(max_clusters):
        inertias = []
        silhouettes = []

        for i in range(2, max_clusters + 1):
            kmeans = KMeans(
                n_clusters=i,
                random_state=0,
                n_init=10
            )
            labels = kmeans.fit_predict(spotify_clustering_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(
                silhouette_score(spotify_clustering_scaled, labels, sample_size=6000, random_state=0)
            )
        return inertias, silhouettes

    with st.spinner('Computing K-means metrics...'):
        inertias, silhouettes = calculate_metrics(max_clusters)

    # Create columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Elbow + Silhouette Analysis")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        x = np.arange(2, max_clusters + 1)
        ax1.plot(x, inertias, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Inertia')
        ax1.fill_between(x, inertias, alpha=0.15, color='#1f77b4')
        ax2.plot(x, silhouettes, marker='s', linewidth=2.5, markersize=7, color='#ff7f0e', label='Silhouette')

        ax1.set_title('Elbow Method and Silhouette Coefficient for K-means', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12, color='#1f77b4')
        ax2.set_ylabel('Silhouette Coefficient', fontsize=12, color='#ff7f0e')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(2, max_clusters + 1, 2))
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        st.pyplot(fig)

    with col2:
        st.subheader("Candidate K Values")
        # Recommend best K based on the elbow
        st.metric(
            "Best K (Elbow)",
            "K = 6",
            "Recommended candidate based on elbow analysis"
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("K-means Metrics Table")
        metrics_df = pd.DataFrame({
            'K (Clusters)': range(2, max_clusters + 1),
            'Inertia': [f'{x:,.0f}' for x in inertias],
            'Silhouette': [f'{x:.4f}' for x in silhouettes]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col4:
        st.subheader("About the Elbow Method")
        st.info("""
        **What is the Elbow Method?**
        
        It helps determine the optimal number of clusters by finding the point where adding more clusters doesn't significantly reduce inertia.
        
        **Key Points:**
        - Look for the "elbow" or "knee" in the curve
        - Beyond this point, inertia decreases slowly
        - A good trade-off uses both inertia reduction and silhouette score
        - The best K shown is based on the highest silhouette score
        """)

with tab2:
    st.header("K=6 Metrics & Radar Map")
    st.markdown("Detailed metrics and radar visualization for K=6 clustering.")
    
    # Perform K-means with k=6
    k6 = 6
    kmeans_k6 = KMeans(n_clusters=k6, random_state=0)
    clusters_k6 = kmeans_k6.fit_predict(spotify_clustering_scaled)
    
    # Get cluster labels
    spotify_with_clusters_k6 = spotify_clustering.copy()
    spotify_with_clusters_k6['cluster'] = clusters_k6
    
    # Cluster sizes
    cluster_sizes_k6 = spotify_with_clusters_k6['cluster'].value_counts().sort_index()
    
    # Metrics for K=6
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Clusters", f"{k6}")
    with col2:
        sil_score_k6 = silhouette_score(spotify_clustering_scaled, clusters_k6)
        st.metric("Silhouette Score", f"{sil_score_k6:.4f}")
    with col3:
        st.metric("Inertia", f"{kmeans_k6.inertia_:,.0f}")
    
    st.divider()
    
    # Cluster sizes
    st.subheader("Cluster Distribution")
    sizes_df_k6 = pd.DataFrame({
        'Cluster': cluster_sizes_k6.index,
        'Number of Observations': cluster_sizes_k6.values,
        'Percentage': [f"{(c/len(clusters_k6))*100:.1f}%" for c in cluster_sizes_k6.values]
    })
    st.dataframe(sizes_df_k6, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Radar map for cluster comparison
    st.subheader("Radar Map - Cluster Comparison")
    
    # Get centroids in original scale
    centroids_scaled_k6 = kmeans_k6.cluster_centers_
    centroids_original_k6 = scaler.inverse_transform(centroids_scaled_k6)
    
    # Feature names for radar
    feature_names = spotify_clustering.columns.tolist()
    
    # Normalize for radar chart (0-1 scale for visualization)
    centroids_df_k6 = pd.DataFrame(centroids_original_k6, columns=feature_names)
    centroids_df_k6.index = [f'Cluster {i}' for i in range(k6)]
    
    # Normalize each feature to 0-1 for radar
    centroids_normalized = (centroids_df_k6 - centroids_df_k6.min()) / (centroids_df_k6.max() - centroids_df_k6.min())
    
    # Create radar chart
    fig_radar, ax_radar = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Number of variables
    num_vars = len(feature_names)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Plot each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, k6))
    
    for idx, (cluster_name, row) in enumerate(centroids_normalized.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # Complete the loop
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=cluster_name, color=colors[idx])
        ax_radar.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Set the labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feature_names, fontsize=10)
    ax_radar.set_title('Radar Map: Cluster Comparison (K=6)', fontsize=16, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize='small')
    
    st.pyplot(fig_radar)
    
    st.divider()
    
    # Centroids table
    st.subheader("Cluster Centroids (Original Scale)")
    st.dataframe(centroids_df_k6, use_container_width=True)
    
    # Feature comparison plot
    st.subheader("Feature Comparison")
    fig_comparison, ax_comparison = plt.subplots(figsize=(12, 6))
    for idx, (cluster_name, row) in enumerate(centroids_df_k6.iterrows()):
        ax_comparison.plot(feature_names, row.values, marker='o', label=cluster_name, color=colors[idx])
    
    ax_comparison.set_xlabel('Features', fontsize=12)
    ax_comparison.set_ylabel('Value', fontsize=12)
    ax_comparison.set_title('Feature Values by Cluster (K=6)', fontsize=16, fontweight='bold')
    ax_comparison.set_xticks(range(len(feature_names)))
    ax_comparison.set_xticklabels(feature_names, rotation=45, ha='right')
    ax_comparison.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize='small')
    ax_comparison.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig_comparison)

with tab3:
    st.header("Cluster Analysis")
    st.markdown("Choose the number of clusters (k) and analyze the resulting clusters.")
    
    # User input for k
    k = st.slider("Select number of clusters (k):", min_value=2, max_value=20, value=2, step=1)
    
    # Perform K-means (matching notebook parameters)
    kmeans = KMeans(n_clusters=k, random_state=0)
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
    
    # Silhouette score
    sil_score = silhouette_score(spotify_clustering_scaled, clusters)
    st.metric("Silhouette Score", f"{sil_score:.4f}")

with tab4:
    st.header("Song Search System")
    st.markdown("Search for a song by name and artist to see which cluster it belongs to.")
    
    # Define feature labels for display
    FEATURE_LABELS = {
        'popularity': 'Popularity',
        'danceability': 'Danceability',
        'energy': 'Energy',
        'liveness': 'Liveness',
        'valence': 'Valence',
        'tempo': 'Tempo',
        'log_speechiness': 'Speechiness',
        'log_instrumentalness': 'Instrumentalness',
        'log_duration_ms': 'Duration'
    }
    
    # Radar features for display
    radar_feats = ['popularity', 'danceability', 'energy', 'liveness', 'valence', 'tempo']
    
    # Load original data for search
    @st.cache_data
    def load_original_data():
        df = pd.read_csv("dataset.csv")
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df.drop(df[df['artists'].isna()].index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    df_original = load_original_data()
    
    # Perform K-means with k=6
    kmeans_search = KMeans(n_clusters=6, random_state=0)
    clusters_search = kmeans_search.fit_predict(spotify_clustering_scaled)
    
    # Create dataframe with clusters
    df_with_clusters = spotify_clustering.copy()
    df_with_clusters['cluster'] = clusters_search
    df_with_clusters['name'] = df_original['track_name'].values
    df_with_clusters['artists'] = df_original['artists'].values
    
    # ── Track explorer ────────────────────────────────────────────────────
    st.subheader("🔍 Track Explorer")
    st.markdown("Search any track name or artist to see which cluster it belongs to and how it compares to the centroid.")

    search = st.text_input("Search by track name or artist", placeholder="e.g. Dua Lipa / Blinding Lights")
    if search:
        mask = (
            df_with_clusters["name"].str.contains(search, case=False, na=False) |
            df_with_clusters["artists"].str.contains(search, case=False, na=False)
        )
        results = df_with_clusters[mask][["name", "artists", "cluster"] + radar_feats].head(20)
        if len(results):
            results["cluster"] = results["cluster"].apply(
                lambda c: f"Cluster {c}"
            )
            st.dataframe(results.rename(columns=FEATURE_LABELS | {"name": "Track", "artists": "Artists", "cluster": "Cluster"}),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("No tracks found. Try a different search term.")

with tab5:
    st.header("Self-Organizing Map (SOM)")
    st.markdown("Use a SOM grid to organize the Spotify tracks by similarity and visualize neuron assignments.")

    som_rows = st.slider("SOM grid rows:", min_value=5, max_value=20, value=10, step=1)
    som_cols = st.slider("SOM grid columns:", min_value=5, max_value=20, value=10, step=1)
    num_iterations = st.slider("Training iterations:", min_value=10, max_value=300, value=120, step=10)
    
    # Check if parameters have changed since last training
    params_changed = (
        'som_rows' not in st.session_state or st.session_state.som_rows != som_rows or
        'som_cols' not in st.session_state or st.session_state.som_cols != som_cols or
        'num_iterations' not in st.session_state or st.session_state.num_iterations != num_iterations
    )
    
    if params_changed and 'som_trained' in st.session_state:
        # Clear previous training results if parameters changed
        del st.session_state.som_trained
        if 'som' in st.session_state:
            del st.session_state.som
        if 'som_labels' in st.session_state:
            del st.session_state.som_labels
        if 'som_rows' in st.session_state:
            del st.session_state.som_rows
        if 'som_cols' in st.session_state:
            del st.session_state.som_cols
        if 'num_iterations' in st.session_state:
            del st.session_state.num_iterations
    
    train_som = st.button("Train SOM")
    
    # Show current parameter status
    if 'som_trained' in st.session_state and st.session_state.som_trained:
        st.success(f"✅ SOM trained with {st.session_state.som_rows}×{st.session_state.som_cols} grid, {st.session_state.num_iterations} iterations")
    elif params_changed:
        st.info("ℹ️ Parameters changed. Click 'Train SOM' to train with new settings.")
    else:
        st.info("ℹ️ Adjust parameters and click 'Train SOM' to start training.")

    if train_som:
        with st.spinner('Training SOM...'):
            som, mapped, som_labels = train_som_cached(som_rows, som_cols, num_iterations, spotify_clustering_scaled)
            
            # Store in session state
            st.session_state.som_trained = True
            st.session_state.som = som
            st.session_state.som_labels = som_labels
            st.session_state.som_rows = som_rows
            st.session_state.som_cols = som_cols
            st.session_state.num_iterations = num_iterations

    # Display visualizations if SOM is trained
    if 'som_trained' in st.session_state and st.session_state.som_trained:
        som = st.session_state.som
        som_labels = st.session_state.som_labels
        som_rows = st.session_state.som_rows
        som_cols = st.session_state.som_cols
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distance Map (U-Matrix)")
            umatrix = compute_u_matrix_cached(som)
            fig_umatrix, ax_umatrix = plt.subplots(figsize=(8, 7))
            im_u = ax_umatrix.imshow(umatrix, cmap='viridis', origin='lower', aspect='auto')
            ax_umatrix.set_title('SOM U-Matrix (Distance Map)', fontsize=13, fontweight='bold')
            ax_umatrix.set_xlabel('Grid Column', fontsize=11)
            ax_umatrix.set_ylabel('Grid Row', fontsize=11)
            cbar_u = plt.colorbar(im_u, ax=ax_umatrix, label='Neighbor Distance')
            plt.tight_layout()
            st.pyplot(fig_umatrix)
        
        with col2:
            st.subheader("Neuron Population Grid")
            som_map = compute_population_grid_cached(som_labels, som_rows, som_cols)
            fig_pop, ax_pop = plt.subplots(figsize=(8, 7))
            im_pop = ax_pop.imshow(som_map, cmap='YlOrRd', origin='lower', aspect='auto')
            ax_pop.set_title('SOM Data Point Distribution', fontsize=13, fontweight='bold')
            ax_pop.set_xlabel('Grid Column', fontsize=11)
            ax_pop.set_ylabel('Grid Row', fontsize=11)
            cbar_pop = plt.colorbar(im_pop, ax=ax_pop, label='Sample Count')
            
            # Add text annotations with adaptive coloring
            for i in range(som_rows):
                for j in range(som_cols):
                    count = int(som_map[i, j])
                    text_color = 'white' if count > som_map.max() / 2 else 'black'
                    ax_pop.text(j, i, count, ha="center", va="center", 
                               color=text_color, fontsize=7, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_pop)
        
        st.divider()
        assignment_counts = compute_assignment_counts_cached(som_labels)
        most_populated = assignment_counts.nlargest(5)
        st.write("**Top 5 Most Populated Neurons:**")
        for rank, (neuron_id, count) in enumerate(most_populated.items(), 1):
            st.metric(f"#{rank}", f"Neuron {int(neuron_id)}", f"{count} samples")


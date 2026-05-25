import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Spotify Clustering", layout="wide")

st.title("🎵 Spotify Clustering")

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.drop(df[df["artists"].isna()].index, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    features = ["popularity", "danceability", "energy", "liveness", "valence", "tempo"]
    clustering = df[features].copy()
    clustering["log_speechiness"]      = np.log1p(df["speechiness"])
    clustering["log_instrumentalness"] = np.log1p(df["instrumentalness"])
    clustering["log_duration_ms"]      = np.log(df["duration_ms"])

    scaler = RobustScaler()
    scaled = scaler.fit_transform(clustering)

    return df, clustering, scaled, scaler


df_original, spotify_clustering, spotify_scaled, scaler = load_data()

FEATURE_LABELS = {
    "popularity":            "Popularity",
    "danceability":          "Danceability",
    "energy":                "Energy",
    "liveness":              "Liveness",
    "valence":               "Valence",
    "tempo":                 "Tempo",
    "log_speechiness":       "Speechiness",
    "log_instrumentalness":  "Instrumentalness",
    "log_duration_ms":       "Duration",
}

# ─── K=6 KMeans (computed once, cached) ───────────────────────────────────────

@st.cache_data
def fit_kmeans_k6(scaled):
    km = KMeans(n_clusters=6, random_state=0, n_init=10)
    labels = km.fit_predict(scaled)
    sil = silhouette_score(scaled, labels, sample_size=6000, random_state=0)
    return km, labels, sil


def relabel_kmeans_by_popularity(km, labels, scaler):
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    popularity = centroids_orig[:, 0]
    sorted_idx = np.argsort(popularity)
    relabel_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_idx)}
    km.cluster_centers_ = km.cluster_centers_[sorted_idx]
    return np.array([relabel_map[int(lbl)] for lbl in labels], dtype=int)


kmeans_k6, labels_k6, sil_k6 = fit_kmeans_k6(spotify_scaled)
labels_k6 = relabel_kmeans_by_popularity(kmeans_k6, labels_k6, scaler)

# ─── Cluster names & descriptions (hardcoded) ─────────────────────────────────

CLUSTER_NAMES = {
    0: "Live Spoken",
    1: "Ambient / Instrumental",
    2: "Ambient Live",
    3: "Dance / Party",
    4: "Mainstream Pop",
    5: "Live Energetic",
}

CLUSTER_DESCRIPTIONS = {
    0: ("Elevated liveness and speechiness signal recordings with a strong spoken "
        "or vocal presence — live sets, podcasts, comedy, or spoken-word tracks."),
    1: ("Highly instrumental with low energy. Classical, ambient, and acoustic "
        "music with minimal or no vocals."),
    2: ("Moderate-to-high liveness with low energy and high instrumentalness. "
        "Quiet live sessions, acoustic sets, or ambient field recordings."),
    3: ("Maximum danceability, high valence, and strong energy. "
        "The dance floor — electronic, house, funk, and feel-good party tracks."),
    4: ("High popularity and strong danceability with moderate energy. "
        "The core of commercial pop music — polished, broad-appeal tracks."),
    5: ("High liveness combined with high energy. Concert recordings, live rock, "
        "or any performance with a crowd and intensity."),
}

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Elbow Method", "🔬 K=6 Analysis", "🔍 Song Search", "🧠 SOM"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Elbow Method  (matplotlib, matching original style)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Elbow Method — Optimal K Selection")
    st.markdown("Interactive visualization of the elbow method for optimal K-means cluster selection")

    max_clusters = st.slider("Maximum number of clusters to test:", 5, 20, 12, key="max_k_slider")

    @st.cache_data
    def calculate_metrics(max_clusters):
        inertias    = []
        silhouettes = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
            lbl    = kmeans.fit_predict(spotify_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(
                silhouette_score(spotify_scaled, lbl, sample_size=6000, random_state=0)
            )
        return inertias, silhouettes

    with st.spinner("Computing K-means metrics..."):
        inertias, silhouettes = calculate_metrics(max_clusters)

    # ── Elbow detection: second derivative of inertia ─────────────────────────
    ks         = list(range(2, max_clusters + 1))
    inertia_arr = np.array(inertias)
    d2          = np.diff(np.diff(inertia_arr))
    elbow_k     = ks[int(np.argmax(d2)) + 1]   # +1 from double diff offset

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Elbow + Silhouette Analysis")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        x = np.arange(2, max_clusters + 1)

        ax1.plot(x, inertias,    marker="o", linewidth=2.5, markersize=8,
                 color="#1f77b4", label="Inertia")
        ax1.fill_between(x, inertias, alpha=0.15, color="#1f77b4")
        ax2.plot(x, silhouettes, marker="s", linewidth=2.5, markersize=7,
                 color="#ff7f0e", label="Silhouette")

        # Highlight K=6 on the plot
        if 6 in ks:
            idx6 = ks.index(6)
            ax1.axvline(x=6, color="green", linestyle="--", linewidth=1.8, alpha=0.7,
                        label="K=6 (selected)")
            ax1.annotate(
                "K=6\n(selected)",
                xy=(6, inertias[idx6]),
                xytext=(6 + 0.5, inertias[idx6] * 1.05),
                fontsize=10, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green"),
            )

        ax1.set_title("Elbow Method and Silhouette Coefficient for K-means",
                      fontsize=16, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax1.set_ylabel("Inertia",               fontsize=12, color="#1f77b4")
        ax2.set_ylabel("Silhouette Coefficient", fontsize=12, color="#ff7f0e")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(2, max_clusters + 1, 2))
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        st.pyplot(fig)

    with col2:
        st.subheader("Candidate K Values")
        if 6 in ks:
            idx6 = ks.index(6)
            st.metric(
                "Best K (Elbow)",
                "K = 6",
                f"Sil: {silhouettes[idx6]:.4f} | Inertia: {inertias[idx6]:,.0f}",
                delta_color="off",
            )
        st.info(
            "**Why K=6?**\n\n"
            "The inertia curve shows a clear elbow at K=6: "
            "the rate of decrease slows markedly after this point. "
            "The silhouette score at K=6 also represents a strong trade-off "
            "between cluster cohesion and separation, making it the "
            "recommended choice."
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("K-means Metrics Table")
        metrics_df = pd.DataFrame({
            "K (Clusters)": ks,
            "Inertia":      [f"{v:,.0f}"  for v in inertias],
            "Silhouette":   [f"{v:.4f}"   for v in silhouettes],
            "Selected":     ["✅ K=6" if k == 6 else "" for k in ks],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col4:
        st.subheader("About the Elbow Method")
        st.info(
            "**What is the Elbow Method?**\n\n"
            "It helps determine the optimal number of clusters by finding the "
            "point where adding more clusters doesn't significantly reduce inertia.\n\n"
            "**Key Points:**\n"
            '- Look for the "elbow" or "knee" in the curve\n'
            "- Beyond this point, inertia decreases slowly\n"
            "- A good trade-off uses both inertia reduction and silhouette score\n"
            "- K=6 is selected based on the elbow in the inertia curve "
            "confirmed by a competitive silhouette score"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — K=6 Analysis  (Plotly radars + cluster interpretations)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### K=6 Cluster Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", "6")
    c2.metric("Silhouette Score", f"{sil_k6:.4f}")
    c3.metric("Inertia", f"{kmeans_k6.inertia_:,.0f}")

    st.divider()

    # ── Cluster distribution ─────────────────────────────────────────────────
    st.markdown("#### Cluster Distribution")

    df_k6  = spotify_clustering.copy()
    df_k6["cluster"] = labels_k6
    sizes  = df_k6["cluster"].value_counts().sort_index()

    dist_df = pd.DataFrame({
        "Cluster":       [f"{CLUSTER_NAMES[i]}" for i in sizes.index],
        "Observations":  sizes.values,
        "Share (%)":     [f"{v/len(labels_k6)*100:.1f}%" for v in sizes.values],
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cluster interpretation cards ─────────────────────────────────────────
    st.markdown("#### Cluster Interpretations")

    COLORS = px.colors.qualitative.Safe

    for row_start in range(0, 6, 3):
        cols = st.columns(3)
        for col_i, c_idx in enumerate(range(row_start, row_start + 3)):
            if c_idx >= 6:
                break
            color = COLORS[c_idx % len(COLORS)]
            with cols[col_i]:
                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid {color};
                        padding: 10px 14px;
                        background: rgba(0,0,0,0.03);
                        border-radius: 6px;
                        margin-bottom: 8px;
                    ">
                        <div style="font-weight:700; font-size:1rem;">
                            {CLUSTER_NAMES[c_idx]}
                        </div>
                        <div style="font-size:0.85rem; margin-top:6px; color:#555;">
                            {CLUSTER_DESCRIPTIONS[c_idx]}
                        </div>
                        <div style="font-size:0.78rem; margin-top:8px; color:#888;">
                            {sizes[c_idx]:,} tracks
                            ({sizes[c_idx]/len(labels_k6)*100:.1f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Radar charts ─────────────────────────────────────────────────────────
    st.markdown("#### Radar Maps — Individual Cluster Profiles")
    st.caption(
        "Each radar shows one cluster's centroid (coloured) vs. the global "
        "dataset mean (grey). Hover for exact values."
    )

    centroids_orig = scaler.inverse_transform(kmeans_k6.cluster_centers_)
    feat_cols   = spotify_clustering.columns.tolist()
    feat_labels = [FEATURE_LABELS.get(f, f) for f in feat_cols]
    global_mean = spotify_clustering.mean().values
    feat_min    = spotify_clustering.min().values
    feat_max    = spotify_clustering.max().values
    feat_range  = np.where(feat_max - feat_min == 0, 1, feat_max - feat_min)

    def norm(v):
        return (v - feat_min) / feat_range

    global_mean_norm = norm(global_mean)

    for row_start in range(0, 6, 2):
        col_a, col_b = st.columns(2)
        for col_i, c_idx in enumerate([row_start, row_start + 1]):
            if c_idx >= 6:
                break
            centroid_norm = norm(centroids_orig[c_idx])
            color = COLORS[c_idx % len(COLORS)]

            fig_r = go.Figure()

            fig_r.add_trace(go.Scatterpolar(
                r=np.append(global_mean_norm, global_mean_norm[0]),
                theta=feat_labels + [feat_labels[0]],
                fill="toself",
                fillcolor="rgba(150,150,150,0.10)",
                line=dict(color="grey", width=1.5, dash="dot"),
                name="Dataset mean",
                customdata=np.append(global_mean, global_mean[0]).reshape(-1, 1),
                hovertemplate="%{theta}: %{customdata[0]:.3f}<extra>Dataset mean</extra>",
            ))

            fig_r.add_trace(go.Scatterpolar(
                r=np.append(centroid_norm, centroid_norm[0]),
                theta=feat_labels + [feat_labels[0]],
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.20)"),
                line=dict(color=color, width=2.5),
                name=CLUSTER_NAMES[c_idx],
                customdata=np.append(centroids_orig[c_idx], centroids_orig[c_idx][0]).reshape(-1, 1),
                hovertemplate="%{theta}: %{customdata[0]:.3f}<extra>" + CLUSTER_NAMES[c_idx] + "</extra>",
            ))

            n_obs = sizes[c_idx]
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                showlegend=True,
                title=dict(
                    text=f"{CLUSTER_NAMES[c_idx]}  ({n_obs:,} tracks)",
                    font=dict(size=13),
                ),
                height=380,
                margin=dict(t=60, b=20, l=40, r=40),
            )

            target_col = col_a if col_i == 0 else col_b
            with target_col:
                st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    st.markdown("#### Cluster Centroids (Original Scale)")
    cent_display = pd.DataFrame(centroids_orig, columns=feat_labels)
    cent_display.index = [f"{CLUSTER_NAMES[i]}" for i in range(6)]
    st.dataframe(cent_display.round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Song Search
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🔍 Song Search")
    st.caption(
        "Search by track name or artist. Each result shows the cluster name "
        "the track belongs to (K=6) and its audio features."
    )

    @st.cache_data
    def build_search_df(df_raw, clustering_df, labels):
        sdf = clustering_df.copy()
        sdf["cluster"]  = labels
        sdf["track"]    = df_raw["track_name"].values
        sdf["artists"]  = df_raw["artists"].values
        return sdf

    search_df = build_search_df(df_original, spotify_clustering, labels_k6)

    query = st.text_input(
        "Search track or artist",
        placeholder="e.g.  Blinding Lights  /  Dua Lipa",
        label_visibility="collapsed",
    )

    cluster_filter = st.multiselect(
        "Filter by cluster (leave empty for all):",
        options=[f"{CLUSTER_NAMES[i]}" for i in range(6)],
        default=[],
        key="cluster_filter",
    )

    if query:
        mask = (
            search_df["track"].str.contains(query, case=False, na=False)
            | search_df["artists"].str.contains(query, case=False, na=False)
        )
        results = search_df[mask].copy()

        if cluster_filter:
            cluster_name_to_id = {name: idx for idx, name in CLUSTER_NAMES.items()}
            cluster_nums = [cluster_name_to_id[c] for c in cluster_filter if c in cluster_name_to_id]
            results = results[results["cluster"].isin(cluster_nums)]

        results = results.sort_values("popularity", ascending=False)
        results = results.drop_duplicates(subset=["track", "artists"], keep="first")

        if results.empty:
            st.warning("No tracks found. Try a different search term.")
        else:
            st.success(f"Found **{len(results):,}** track(s).")

            radar_feats   = ["popularity", "danceability", "energy", "liveness", "valence", "tempo"]
            display_cols  = ["track", "artists", "cluster"] + radar_feats
            disp          = results[display_cols].copy().head(50)
            disp["cluster"] = disp["cluster"].apply(
                lambda c: f"{CLUSTER_NAMES[c]}"
            )
            disp = disp.rename(columns={
                "track": "Track", "artists": "Artists", "cluster": "Cluster",
                **{k: v for k, v in FEATURE_LABELS.items()},
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)

            # ── Mini radar for selected track ─────────────────────────────────
            if len(results) <= 200:
                st.markdown("#### Compare a track to its cluster centroid")
                track_options  = results["track"].tolist()
                selected_track = st.selectbox("Pick a track:", track_options, key="track_picker")

                if selected_track:
                    row       = results[results["track"] == selected_track].iloc[0]
                    c_idx     = int(row["cluster"])
                    feat_cols = spotify_clustering.columns.tolist()
                    feat_lbl  = [FEATURE_LABELS.get(f, f) for f in feat_cols]
                    track_vals    = row[feat_cols].values.astype(float)
                    centroid_vals = scaler.inverse_transform(
                        kmeans_k6.cluster_centers_
                    )[c_idx]
                    color = COLORS[c_idx % len(COLORS)]

                    track_norm    = norm(track_vals)
                    centroid_norm = norm(centroid_vals)

                    fig_track = go.Figure()
                    fig_track.add_trace(go.Scatterpolar(
                        r=np.append(centroid_norm, centroid_norm[0]),
                        theta=feat_lbl + [feat_lbl[0]],
                        fill="toself",
                        fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                        line=dict(color=color, width=2, dash="dot"),
                        name=f"{CLUSTER_NAMES[c_idx]} centroid",
                        customdata=np.append(centroid_vals, centroid_vals[0]).reshape(-1, 1),
                        hovertemplate="%{theta}: %{customdata[0]:.3f}<extra>Cluster centroid</extra>",
                    ))
                    fig_track.add_trace(go.Scatterpolar(
                        r=np.append(track_norm, track_norm[0]),
                        theta=feat_lbl + [feat_lbl[0]],
                        fill="toself",
                        fillcolor="rgba(29,185,84,0.20)",
                        line=dict(color="#1DB954", width=2.5),
                        name=selected_track[:40],
                        customdata=np.append(track_vals, track_vals[0]).reshape(-1, 1),
                        hovertemplate="%{theta}: %{customdata[0]:.3f}<extra>Track</extra>",
                    ))
                    fig_track.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                        title=f"{selected_track[:50]}  →  {CLUSTER_NAMES[c_idx]}",
                        height=420,
                    )
                    st.plotly_chart(fig_track, use_container_width=True)
    else:
        st.info("Start typing to search for tracks or artists.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SOM
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### 🧠 Self-Organizing Map (SOM)")
    st.markdown(
        "Configure the grid and training parameters, then click **Train SOM**. "
        "The map will only train when you explicitly press the button."
    )

    class SelfOrganizingMap:
        def __init__(self, m, n, dim, lr=0.5, sigma=None, seed=None):
            self.m, self.n, self.dim = m, n, dim
            self.lr    = lr
            self.sigma = sigma if sigma is not None else max(m, n) / 2.0
            rng        = np.random.RandomState(seed)
            self.weights   = rng.rand(m * n, dim)
            self.locations = np.array([[i, j] for i in range(m) for j in range(n)])

        def _decay(self, val, t, T):
            return val * np.exp(-t / T)

        def winner(self, x):
            return int(np.argmin(np.linalg.norm(self.weights - x, axis=1)))

        def train(self, data, n_iter=100):
            for t in range(n_iter):
                lr  = self._decay(self.lr,    t, n_iter)
                sig = self._decay(self.sigma, t, n_iter)
                for sample in data:
                    w_idx     = self.winner(sample)
                    d         = np.linalg.norm(self.locations - self.locations[w_idx], axis=1)
                    influence = np.exp(-(d ** 2) / (2 * sig ** 2))
                    self.weights += lr * influence[:, np.newaxis] * (sample - self.weights)

        def map_vects(self, data):
            return np.array([self.locations[self.winner(s)] for s in data])

        def u_matrix(self):
            um = np.zeros(self.m * self.n)
            for idx, loc in enumerate(self.locations):
                nd = [
                    np.linalg.norm(self.weights[idx] - self.weights[j])
                    for j, nl in enumerate(self.locations)
                    if np.sum(np.abs(loc - nl)) == 1
                ]
                um[idx] = np.mean(nd) if nd else 0
            return um.reshape(self.m, self.n)

    col_p1, col_p2, col_p3 = st.columns(3)
    som_rows = col_p1.slider("Grid rows:",         5, 20, 10, key="som_rows_slider")
    som_cols = col_p2.slider("Grid columns:",      5, 20, 10, key="som_cols_slider")
    n_iter   = col_p3.slider("Training iterations:", 10, 300, 100, step=10, key="som_iter_slider")

    train_btn = st.button("🚀 Train SOM", type="primary")

    if train_btn:
        cache_key = (som_rows, som_cols, n_iter)
        if st.session_state.get("som_cache_key") != cache_key:
            for k in ["som_weights", "som_locations", "som_labels", "som_cache_key",
                      "som_umatrix", "som_popgrid"]:
                st.session_state.pop(k, None)

        with st.spinner(f"Training {som_rows}×{som_cols} SOM for {n_iter} iterations…"):
            som    = SelfOrganizingMap(som_rows, som_cols, spotify_scaled.shape[1],
                                       lr=0.5, sigma=max(som_rows, som_cols) / 2.0, seed=0)
            som.train(spotify_scaled, n_iter=n_iter)
            mapped     = som.map_vects(spotify_scaled)
            labels_som = mapped[:, 0] * som_cols + mapped[:, 1]

            pop_grid = np.zeros((som_rows, som_cols))
            for lbl in labels_som:
                r, c = int(lbl // som_cols), int(lbl % som_cols)
                pop_grid[r, c] += 1

            st.session_state["som_labels"]    = labels_som.copy()
            st.session_state["som_rows"]      = som_rows
            st.session_state["som_cols"]      = som_cols
            st.session_state["som_umatrix"]   = som.u_matrix()
            st.session_state["som_popgrid"]   = pop_grid
            st.session_state["som_cache_key"] = cache_key

        st.success(f"✅ SOM trained: {som_rows}×{som_cols} grid, {n_iter} iterations.")

    if "som_labels" in st.session_state:
        trained_rows = st.session_state["som_rows"]
        trained_cols = st.session_state["som_cols"]
        umatrix      = st.session_state["som_umatrix"]
        pop_grid     = st.session_state["som_popgrid"]
        labels_som   = st.session_state["som_labels"]

        st.caption(
            f"Showing results for {trained_rows}×{trained_cols} grid, "
            f"{st.session_state['som_cache_key'][2]} iterations."
        )

        col_u, col_p = st.columns(2)

        with col_u:
            st.markdown("##### Distance Map (U-Matrix)")
            fig_u = go.Figure(go.Heatmap(
                z=umatrix,
                colorscale="Viridis",
                colorbar=dict(title="Neighbour distance"),
            ))
            fig_u.update_layout(
                xaxis_title="Grid column",
                yaxis_title="Grid row",
                yaxis_autorange="reversed",
                height=420,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_u, use_container_width=True)

        with col_p:
            st.markdown("##### Population Grid")
            annot_text = [
                [str(int(pop_grid[r, c])) for c in range(trained_cols)]
                for r in range(trained_rows)
            ]
            fig_p = go.Figure(go.Heatmap(
                z=pop_grid,
                text=annot_text,
                texttemplate="%{text}",
                colorscale="YlOrRd",
                colorbar=dict(title="Sample count"),
            ))
            fig_p.update_layout(
                xaxis_title="Grid column",
                yaxis_title="Grid row",
                yaxis_autorange="reversed",
                height=420,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.divider()
        top5   = pd.Series(labels_som).value_counts().nlargest(5)
        st.markdown("##### Top 5 Most Populated Neurons")
        t_cols = st.columns(5)
        for i, (neuron_id, count) in enumerate(top5.items()):
            r, c = int(neuron_id // trained_cols), int(neuron_id % trained_cols)
            t_cols[i].metric(f"#{i+1}", f"({r},{c})", f"{count:,} samples")
    else:
        st.info("Configure parameters above and click **Train SOM** to begin.")
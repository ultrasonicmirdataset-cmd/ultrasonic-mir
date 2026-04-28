"""
clustering.py
Purpose: Provides various clustering algorithms (Agglomerative, K-Means, Spectral) to group NMF components into meaningful audio sources based on spectral and temporal features.
THERE ARE A LOT OF EXPERIMENTAL FUNCTIONS HERE, THE ONE THAT WAS USED IS cluster_components_spectral.
"""
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew


# ─────────────────────────────────────────────────────────────────────────────
# Shared feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(W, H, sr, n_fft):
    """
    Extracts spectral and temporal features from NMF components.
    Used by the Agglomerative and K-Means clustering methods.

    Returns
    -------
    features_scaled : ndarray, shape (n_components, n_features)
    """
    W_s = np.maximum(W, 1e-10)
    P = W_s ** 2

    mel_spec = librosa.feature.melspectrogram(S=P, sr=sr, n_fft=n_fft, hop_length=1, n_mels=128)
    log_mel  = librosa.power_to_db(mel_spec, ref=np.max)

    feat_mfcc     = librosa.feature.mfcc(S=log_mel, n_mfcc=13, sr=sr)          # (13, K)
    feat_centroid = librosa.feature.spectral_centroid(S=W_s, sr=sr, n_fft=n_fft)  # (1,  K)
    feat_rolloff  = librosa.feature.spectral_rolloff(S=W_s, sr=sr, n_fft=n_fft)   # (1,  K)
    feat_flatness = librosa.feature.spectral_flatness(S=W_s)                       # (1,  K)
    feat_skew     = skew(H, axis=1).reshape(1, -1)                                 # (1,  K)

    # Stack (n_features, K) → transpose → (K, n_features)
    features_stacked = np.vstack([
        feat_mfcc, feat_centroid, feat_rolloff, feat_flatness, feat_skew
    ]).T  # (K, 17)

    # Append full correlation matrix as (K, K) columns — each row i gets
    # component i's Pearson correlation with every other component
    row_correlation = np.corrcoef(H)              # (K, K)
    features = np.hstack([features_stacked, row_correlation])  # (K, 17+K)

    scaler = StandardScaler()
    return scaler.fit_transform(features)


# ─────────────────────────────────────────────────────────────────────────────
# Affinity helpers (used by spectral clustering only)
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_affinity(W, H, sr, n_fft):
    """
    (K x K) affinity based on timbral similarity between spectral bases.
    Uses cosine similarity on MFCC / centroid / rolloff / flatness / skew features.
    Range: [0, 1].
    """
    W_s = np.maximum(W, 1e-10)
    P   = W_s ** 2

    mel_spec = librosa.feature.melspectrogram(S=P, sr=sr, n_fft=n_fft, hop_length=1, n_mels=128)
    log_mel  = librosa.power_to_db(mel_spec, ref=np.max)

    feat_mfcc     = librosa.feature.mfcc(S=log_mel, n_mfcc=13, sr=sr)
    feat_centroid = librosa.feature.spectral_centroid(S=W_s, sr=sr, n_fft=n_fft)
    feat_rolloff  = librosa.feature.spectral_rolloff(S=W_s, sr=sr, n_fft=n_fft)
    feat_flatness = librosa.feature.spectral_flatness(S=W_s)
    feat_skew     = skew(H, axis=1).reshape(1, -1)

    features = np.vstack([
        feat_mfcc, feat_centroid, feat_rolloff, feat_flatness, feat_skew
    ]).T  # (K, 17)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    sim = cosine_similarity(features_scaled)   # [-1, 1]
    return (sim + 1) / 2                       # [0, 1]


def _temporal_affinity(H):
    """
    (K x K) affinity based on co-activation in time (rows of H).
    Two components that fire together → high affinity, regardless of spectral shape.
    Uses Pearson correlation normalized to [0, 1].
    """
    corr = np.corrcoef(H)       # [-1, 1]
    return (corr + 1) / 2       # [0, 1]


def _build_combined_affinity(W, H, sr, n_fft, temporal_weight):
    """
    Blends spectral and temporal affinities into a single (K x K) matrix.

    Parameters
    ----------
    temporal_weight : float in [0, 1]
        0.0 → pure spectral  |  0.6 → recommended for drums/percussion
        0.5 → balanced       |  1.0 → pure temporal

    Returns
    -------
    affinity   : (K, K) combined matrix
    S_spectral : (K, K) spectral-only component
    S_temporal : (K, K) temporal-only component
    """
    n_fft_W = 2 * (W.shape[0] - 1)
    S_spec = _spectral_affinity(W, H, sr, n_fft_W)
    S_temp = _temporal_affinity(H)

    affinity = (1 - temporal_weight) * S_spec + temporal_weight * S_temp
    affinity = np.clip(affinity, 0, 1)
    affinity = (affinity + affinity.T) / 2      # enforce symmetry
    return affinity, S_spec, S_temp


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 1: Agglomerative Clustering  (spectral features only)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_components(W, H, n_sources, sr=44100, n_fft=1024):
    """
    Groups NMF components into n_sources using Ward Agglomerative Clustering.

    Clusters purely on timbral features (MFCCs, centroid, rolloff, flatness,
    skew, and pairwise Pearson correlation). Good when components from different
    instruments have clearly distinct spectral profiles.

    Parameters
    ----------
    W        : ndarray (n_bins, K)   — NMF spectral bases
    H        : ndarray (K, n_frames) — NMF activations
    n_sources: int                   — target number of sources
    sr       : int                   — sample rate
    n_fft    : int                   — FFT size used to produce W

    Returns
    -------
    W_clustered : (n_bins, n_sources)
    H_clustered : (n_sources, n_frames)
    """
    n_components = W.shape[1]
    if n_sources >= n_components:
        return W, H

    print(f"[Agglomerative] Clustering {n_components} components → {n_sources} sources...")

    features_scaled = extract_features(W, H, sr, n_fft)

    clustering = AgglomerativeClustering(n_clusters=n_sources, linkage='ward')
    labels = clustering.fit_predict(features_scaled)

    W_clustered = np.zeros((W.shape[0], n_sources))
    H_clustered = np.zeros((n_sources, H.shape[1]))

    for cluster_id in range(n_sources):
        comp_indices = np.where(labels == cluster_id)[0]
        print(f"  Source {cluster_id + 1}: components {comp_indices}")
        W_clustered[:, cluster_id] = np.sum(W[:, comp_indices], axis=1)
        H_clustered[cluster_id, :] = np.sum(H[comp_indices, :], axis=0)

    return W_clustered, H_clustered


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 2: K-Means Clustering  (spectral features only)
# ─────────────────────────────────────────────────────────────────────────────

def Source_Clustering(W, H, inst_num=3, sr=44100, n_mfcc=13, print_cluster_stat=False):
    """
    Groups NMF components into inst_num clusters using K-Means.

    Returns per-cluster W and H arrays rather than summing them, which lets the
    caller handle each group independently (e.g. separate reconstruction per source).

    Parameters
    ----------
    W               : ndarray (n_bins, K)
    H               : ndarray (K, n_frames)
    inst_num        : int  — number of instrument clusters
    sr              : int  — sample rate
    print_cluster_stat : bool — print energy per cluster

    Returns
    -------
    bases      : list of K arrays, each (n_bins, k_i)
    activation : list of K arrays, each (k_i, n_frames)
    """
    n_fft = 2 * (W.shape[0] - 1)

    features_scaled = extract_features(W, H, sr, n_fft)

    kmeans = KMeans(n_clusters=inst_num, random_state=42, n_init='auto', max_iter=300)
    labels = kmeans.fit_predict(features_scaled)

    bases      = []
    activation = []

    for cluster_idx in range(inst_num):
        comp_indices = np.where(labels == cluster_idx)[0]
        bases.append(W[:, comp_indices])
        activation.append(H[comp_indices, :])

    if print_cluster_stat:
        for cluster_id in range(inst_num):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > 0:
                cluster_energy = np.sum([np.sum(H[idx, :]**2) for idx in indices])
                print(f"  Instrument {cluster_id+1}: {len(indices)} components, "
                      f"Total activation energy: {cluster_energy:.2f}")

    return bases, activation


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 3: Spectral Clustering on combined affinity  (spectral + temporal)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_components_spectral(W, H, n_sources, sr=44100, temporal_weight=0.6):
    """
    Groups NMF components into n_sources using Spectral Clustering on a
    combined spectral + temporal affinity matrix.

    WHY USE THIS:
        Components from the same instrument often co-activate at the same time
        even when their spectral shapes differ (e.g. kick drum at 60 Hz and
        cymbal crash at 8 kHz both hit on the beat). This method captures that
        relationship by blending:
            - Spectral affinity: are the timbral features similar?
            - Temporal affinity: do the components fire at the same time?

        Spectral Clustering on a graph (vs K-Means in feature space) naturally
        finds groups that are "connected" by co-activation chains, not just
        close in MFCC distance.

    WHEN TO PREFER THIS OVER OPTIONS 1/2:
        - Drum kits or percussion (components span full frequency range)
        - Any source where components are spectrally diverse but temporally linked
        - When K-Means / Agglomerative splits one instrument across multiple clusters

    Parameters
    ----------
    W               : ndarray (n_bins, K)   — NMF spectral bases
    H               : ndarray (K, n_frames) — NMF activations
    n_sources       : int                   — target number of sources
    sr              : int                   — sample rate
    temporal_weight : float in [0, 1]
        Controls the blend of the affinity matrix:
            0.0 → pure timbral similarity  (same as agglomerative, roughly)
            0.5 → balanced
            0.6 → recommended default      (co-activation slightly preferred)
            1.0 → pure temporal            (only timing matters)

    Returns
    -------
    W_clustered : (n_bins, n_sources)
    H_clustered : (n_sources, n_frames)
    """
    n_components = W.shape[1]
    if n_sources >= n_components:
        return W, H

    print(f"[Spectral] Clustering {n_components} components → {n_sources} sources...")
    print(f"  Temporal weight: {temporal_weight:.0%}  |  Spectral weight: {1-temporal_weight:.0%}")

    n_fft = 2 * (W.shape[0] - 1)
    affinity, _, _ = _build_combined_affinity(W, H, sr, n_fft, temporal_weight)

    clustering = SpectralClustering(
        n_clusters=n_sources,
        affinity='precomputed',     # we supply our own (K x K) affinity matrix
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(affinity)

    W_clustered = np.zeros((W.shape[0], n_sources))
    H_clustered = np.zeros((n_sources, H.shape[1]))

    for cluster_id in range(n_sources):
        comp_indices = np.where(labels == cluster_id)[0]
        print(f"  Source {cluster_id + 1}: components {comp_indices}")
        W_clustered[:, cluster_id] = np.sum(W[:, comp_indices], axis=1)
        H_clustered[cluster_id, :] = np.sum(H[comp_indices, :], axis=0)

    return W_clustered, H_clustered
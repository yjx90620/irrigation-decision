"""Environmental distance/similarity analysis across sites (研究方案 6.5).

AWC is dropped before distance computation - it's currently identical
across all 5 sites (same standard soil everywhere), so including it would
just add a zero-variance column and distort standardized-Euclidean
distances if it ever stops being constant without this getting revisited.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FINGERPRINT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "environmental_fingerprints.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

CLIMATE_FEATURES = ["P_mm", "ET0_mm", "aridity_index", "CV_P", "max_dry_spell_days", "hot_days", "GDD", "PCI"]


def main():
    df = pd.read_csv(FINGERPRINT_PATH, index_col="site_id")
    X = StandardScaler().fit_transform(df[CLIMATE_FEATURES])

    dist_matrix = pd.DataFrame(squareform(pdist(X, metric="euclidean")), index=df.index, columns=df.index)
    dist_matrix.to_csv(OUT_DIR / "site_distance_matrix.csv")
    print("=== standardized-Euclidean climate distance matrix ===")
    print(dist_matrix.round(2).to_string())

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    pca_df = pd.DataFrame(pcs, index=df.index, columns=["PC1", "PC2"])
    pca_df.to_csv(OUT_DIR / "site_pca.csv")
    print(f"\n=== PCA (explained variance ratio: {pca.explained_variance_ratio_.round(3)}) ===")
    print(pca_df.round(2).to_string())

    linkage_matrix = linkage(X, method="ward")
    clusters = fcluster(linkage_matrix, t=2, criterion="maxclust")
    cluster_df = pd.DataFrame({"cluster": clusters}, index=df.index)
    cluster_df.to_csv(OUT_DIR / "site_clusters.csv")
    print("\n=== hierarchical clustering (2 clusters, ward linkage) ===")
    print(cluster_df.to_string())

    off_diagonal = dist_matrix.copy()
    np.fill_diagonal(off_diagonal.values, np.nan)
    site_a, site_b = off_diagonal.stack().idxmin()
    print(f"\nmost similar pair: {site_a} <-> {site_b} (distance={off_diagonal.loc[site_a, site_b]:.2f})")


if __name__ == "__main__":
    main()

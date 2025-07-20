import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import ta
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

# Loading Dataset
clus_df = pd.read_csv(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\3.Clustering Task\woo-BTC USDT.csv')
clus_df['timestamp'] = pd.to_datetime(clus_df['timestamp'])
clus_df = clus_df[clus_df['timestamp'].dt.year != 2017]
clus_df['year'] = clus_df['timestamp'].dt.year

clus_df = clus_df.set_index('timestamp')

# Feature Engineering
clus_df['returns'] = clus_df['close'].pct_change().round(2)
clus_df['sma'] = clus_df['close'].shift(1).rolling(window=7).mean().round(2)
clus_df['ema'] = clus_df['close'].shift(1).ewm(span=7, adjust=False).mean().round(2)

bb = ta.volatility.BollingerBands(close=clus_df['close'].shift(1), window=7)
clus_df['bbh'] = bb.bollinger_hband().round(2)
clus_df['bbm'] = bb.bollinger_mavg().round(2)
clus_df['bbl'] = bb.bollinger_lband().round(2)

clus_df['volatility'] = clus_df['close'].shift(1).rolling(window=7).std().round(2)
clus_df['rsi'] = ta.momentum.RSIIndicator(close=clus_df['close'].shift(1), window=14).rsi().round(2)
clus_df['close_mom_7'] = (clus_df['close'].shift(1) / clus_df['close'].shift(8) - 1).round(2)

clus_df.dropna(inplace=True)

clustering_pipeline = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\3.Clustering Task\clustering_pipeline.pkl')

# Predict clusters
clusters = clustering_pipeline.predict(clus_df)
clus_df['cluster'] = clusters

# Load 2D PCA model for visualization
pca_2d = joblib.load(r'C:\Users\KarthikKodam(Quadran\vs\Assignment\assignment-2\3.Clustering Task\pca_2.pkl')

# Get intermediate PCA-transformed data (converts dataframe to 4 features)
X_scaled = clustering_pipeline.named_steps['scaler'].transform(clus_df.drop(columns='cluster'))
X_pca = clustering_pipeline.named_steps['pca'].transform(X_scaled)

# Convert to 2D for visualization
X_pca_2d = pca_2d.transform(X_pca)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_pca)

# --- PCA 2D Plot ---
# plt.figure(figsize=(16, 4))
# sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=clus_df['cluster'], palette='Set1')
# plt.title('Clusters (PCA 2D)')
# plt.tight_layout()
# plt.show()

# --- PCA vs t-SNE ---
fig, axs = plt.subplots(1, 2, figsize=(16, 4))

sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=clus_df['cluster'], palette='Set1', ax=axs[0])
axs[0].set_title('Clusters (PCA 2D)')

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clus_df['cluster'], palette='coolwarm', ax=axs[1])
axs[1].set_title('Clusters (t-SNE)')

plt.tight_layout()
plt.show()

# --- Clustering Evaluation ---
sil_score = silhouette_score(X_pca, clus_df['cluster'])
print(f'Silhouette Score: {sil_score:.4f}')

db_score = davies_bouldin_score(X_pca, clus_df['cluster'])
print(f'Davies-Bouldin Index: {db_score:.4f}')

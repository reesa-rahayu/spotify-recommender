import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# Data Understanding
data = pd.read_csv('data/spotify_dataset.csv')
data.sample(5)
data.info()
data.drop(columns=['Unnamed: 0'], inplace=True)
data.info()
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
num_cols
data.describe()
## Duplicate
# Check duplicates
data.duplicated().sum()
data['track_id'].duplicated().sum()
data['track_id'].nunique()
data['track_name'].duplicated().sum()
## Missing Value
# Check for missing values
data.isnull().sum()
## Outliers
for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data[col], orient='h')
    plt.title(f'Boxplot of {col}', fontsize=20)
    plt.xlabel(col)
    plt.show()
## EDA
### Numerik
num_cols_data = data[num_cols]
num_cols_data.head()
num_cols_data.hist(
    bins=30,
    figsize=(20, 15),
    color='green',
    edgecolor='black'
)
plt.suptitle('Distribution of Numerical Features', fontsize=20)
plt.tight_layout()
plt.show()
data_sorted_by_popularity = data.sort_values(by='popularity', ascending=False)
data_sorted_by_popularity.head()
### Categorical
category_cols = data.select_dtypes(include=['object', 'bool']).columns
category_cols
categoty_cols_data = data[category_cols]
categoty_cols_data.sample(5)
data['track_genre'].unique()
data.groupby('track_genre').size().reset_index(name='count').sort_values(by='count', ascending=False).head(10)
data.groupby('track_genre').size().reset_index(name='count').sort_values(by='count', ascending=True).head(10)
plt.figure(figsize=(20, 6))
plt.title('Distribution of Track Genres')
sns.countplot(data=data, x='track_genre', order=data['track_genre'].value_counts().index, color='green')
plt.xticks(rotation=90)
plt.show()
data['explicit'].value_counts()
### Bivariate
genre_popularity = data.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)
genre_popularity = pd.DataFrame(genre_popularity).reset_index()
genre_popularity.head(10)
plt.figure(figsize=(10, 6))
plt.title('Distribution of Track Popularity')
plt.bar(genre_popularity['track_genre'].head(10), genre_popularity['popularity'].head(10), color='green')
plt.xlabel('Track Genre')
plt.ylabel('Popularity')
plt.xticks(rotation=90)
plt.show()
### Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(num_cols_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.show()
# Data Preparation
## Data Cleaning
### Duplicate
data = data.sort_values('popularity', ascending=False)
data = data.drop_duplicates(subset=['track_id'], keep='first')
data = data.reset_index(drop=True)
data.duplicated(subset=['track_id']).sum()
data.head()
data.shape
data = data.drop_duplicates(subset=['track_name', 'artists'], keep='first')
data = data.reset_index(drop=True)
data.duplicated(subset=['track_name', 'artists']).sum()
data.shape
### Missing Value
data = data.dropna()
data.shape
### Outlier
def convert_to_minutes(miliseconds):
    seconds = miliseconds / 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes)}:{int(seconds):02d}"
average_duration = data['duration_ms'].mean()
print(f"Average duration of tracks: {average_duration} ms or {convert_to_minutes(average_duration)}")
min_duration = data['duration_ms'].min()
print(f"Minimum duration of tracks: {min_duration} ms or {convert_to_minutes(min_duration)}")
max_duration = data['duration_ms'].max()
print(f"Maximum duration of tracks: {max_duration} ms or {convert_to_minutes(max_duration)}")
data.sort_values('duration_ms', ascending=False).head(10)
data.sort_values('duration_ms', ascending=True).head(10)
upper_bound = 600000
lower_bound = 45000

# Filter the data 
data_filtered = data[(data['duration_ms'] >= lower_bound) & (data['duration_ms'] <= upper_bound)]
data_filtered.describe()
data = data_filtered
data.shape
data['explicit'] = data['explicit'].astype(int)
data['explicit'].sample(5)
## Normalization
scaler = MinMaxScaler()
data_scaled = data.copy()
# Normalize numerical columns
data_scaled[num_cols] = scaler.fit_transform(data[num_cols])
data_scaled[num_cols].describe()
# PCA
selected_cols = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo', 'time_signature']
music_features_data = data_scaled[selected_cols]
music_features_data.sample(5)
from sklearn.decomposition import PCA
n_components = 0.95
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(music_features_data)

print(f"Original feature shape: {music_features_data.shape}")
print(f"PCA feature shape: {pca_features.shape}")
# Modelling
output_columns = ['track_id', 'track_name', 'artists', 'popularity', 'track_genre']
## Popularity Based Recommendations
def popularity_recommendation(input_song_name, num_recommendations=5):
    #check if the song exists in the dataset
    if input_song_name not in data['track_name'].values:
        raise ValueError(f"The song '{input_song_name}' is not in the dataset.")

    # Get the popularity of the given track
    track_popularity = data.loc[data['track_name'] == input_song_name, 'popularity'].values[0]
    
    # Get the most popular tracks
    most_popular_tracks = data[data['popularity'] >= track_popularity].sort_values(by='popularity', ascending=False)
    
    # Return the top 10 most popular tracks
    df =  most_popular_tracks[output_columns].head(num_recommendations)
    ids = df['track_id'].tolist()
    
    return df, ids
## KNN Based Recommendations
from sklearn.neighbors import NearestNeighbors
def knn_based_recommendations(input_song_name, num_recommendations=5):
    #check if the song exists in the dataset
    if input_song_name not in data['track_name'].values:
        raise ValueError(f"The song '{input_song_name}' is not in the dataset.")
    
    knn_model = NearestNeighbors(n_neighbors=num_recommendations, metric='cosine')
    knn_model.fit(pca_features)

    # Get the index of the input song
    input_song_index = data[data['track_name'] == input_song_name].index[0]
    input_song_vector = pca_features[input_song_index].reshape(1, -1)
    
    distances, indices = knn_model.kneighbors(input_song_vector, n_neighbors=num_recommendations + 10)
    similar_song_indices = indices[0][1:]  # Exclude the first index (the song itself)
    
    knn_based_recommendations = data.iloc[similar_song_indices][output_columns]
    knn_based_recommendations['distance'] = distances[0][1:]  # Exclude the first distance (the song itself)

    # Delete duplicate songs
    knn_based_recommendations.drop_duplicates(subset=['track_name'])
    knn_based_recommendations.sort_values(by='distance', ascending=True)

    df = knn_based_recommendations.head(num_recommendations)
    ids = df['track_id'].tolist()
    
    return df, ids
## Cluster Based Recommendation
from sklearn.cluster import KMeans
wcss = []
for k in range(5, 51, 5):  # Try from 5 to 50 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pca_features) 
    wcss.append(kmeans.inertia_)

plt.plot(range(5, 51, 5), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(pca_features)
data['cluster'] = clusters
data[['track_name', 'artists', 'cluster']].sample(5)
data['cluster'].value_counts()
data[data['cluster'] == 0].track_genre.value_counts().head(10)
def cluster_based_recommendations(input_song_name, num_recommendations=5):
    # Get the cluster of the given track
    track_cluster = data.loc[data['track_name'] == input_song_name, 'cluster'].values[0]

    cluster_tracks = data[data['cluster'] == track_cluster].sort_values(by='popularity', ascending=False)
    cluster_tracks = cluster_tracks[cluster_tracks['track_name'] != input_song_name]
    df = cluster_tracks[output_columns].head(num_recommendations)
    ids = df['track_id'].tolist()

    return df, ids

## Hybrid Recommendation
def hybrid_recommendation(input_song_name, num_recommendations=5):
    # Check if the song exists in the dataset
    if input_song_name not in data['track_name'].values:
        raise ValueError(f"The song '{input_song_name}' is not in the dataset.")

    # KNN model for similarity
    knn_model = NearestNeighbors(n_neighbors=num_recommendations + 10, metric='cosine')
    knn_model.fit(pca_features)

    # Get the index of the input song
    input_song_index = data[data['track_name'] == input_song_name].index[0]
    input_song_vector = pca_features[input_song_index].reshape(1, -1)

    distances, indices = knn_model.kneighbors(input_song_vector, n_neighbors=num_recommendations + 10)
    similar_song_indices = indices[0][1:]  # Exclude the first index (the song itself)

    knn_based_recommendations = data.iloc[similar_song_indices].copy()
    knn_based_recommendations['distance'] = distances[0][1:]  # Exclude the first distance (the song itself)

    # Filter based on input song cluster
    input_song_cluster = data.loc[input_song_index, 'cluster']
    candidate_songs = knn_based_recommendations[knn_based_recommendations['cluster'] == input_song_cluster].copy()

    # If no songs are found in the cluster, use the unfiltered recommendations
    if candidate_songs.empty:
        candidate_songs = knn_based_recommendations.copy()  # Use a copy to avoid modifying the original

    # Sort by popularity and similarity (hybrid ranking)
    candidate_songs['hybrid_score'] = 0.6 * candidate_songs['distance'] + 0.4 * (
            candidate_songs['popularity'] / 100)
    candidate_songs = candidate_songs.sort_values(by='hybrid_score', ascending=False)
    
    # Get the top recommendations
    top_recommendations = candidate_songs[output_columns].head(num_recommendations)
    ids = top_recommendations['track_id'].tolist()
    

    return top_recommendations, ids
# Evaluation
### define ground truth by genre
def get_ground_truth_by_genre(df, input_track_id):
    genre = df[df['track_id'] == input_track_id]['track_genre'].values[0]
    df = df[df['track_genre'] == genre][output_columns]
    ids = df['track_id'].tolist()
    return df, ids
genre_truth, genre_truth_ids = get_ground_truth_by_genre(data, input_song_id)
genre_truth
## Scoring
from sklearn.metrics import ndcg_score
def precision_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    relevant = set(y_true)
    return len([i for i in y_pred if i in relevant]) / k

def recall_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    relevant = set(y_true)
    return len([i for i in y_pred if i in relevant]) / len(relevant) if relevant else 0

def f1_score_at_k(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0

def average_precision(y_true, y_pred, k):
    score = 0.0
    hits = 0.0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(y_true), k) if y_true else 0

def evaluate_all(y_true, y_pred, k=5):
    p = precision_at_k(y_true, y_pred, k)
    r = recall_at_k(y_true, y_pred, k)
    f1 = f1_score_at_k(p, r)
    ap = average_precision(y_true, y_pred, k)

    # NDCG assumes relevance scores → binary (1 if relevant)
    true_relevance = [[1 if i in y_true else 0 for i in y_pred[:k]]]
    predicted_scores = [[1.0 / (i + 1) for i in range(k)]]
    ndcg = ndcg_score(true_relevance, predicted_scores)

    return {
        'Precision@{}'.format(k): round(p, 4),
        'Recall@{}'.format(k): round(r, 4),
        'F1@{}'.format(k): round(f1, 4),
        'MAP@{}'.format(k): round(ap, 4),
        'NDCG@{}'.format(k): round(ndcg, 4)
    }
## Blinding Light by The Weekend
input_song_name = 'Blinding Lights'
song = data.loc[data['track_name'] == input_song_name]
song = song[0:1]
song
input_song_id = song['track_id'].iloc[0]
input_song_id
pop_pred, pop_ids = popularity_recommendation(input_song_name, num_recommendations=5)
pop_pred
knn_pred, knn_ids = knn_based_recommendations(input_song_name, num_recommendations=5)
knn_pred
cluster_pred, cluster_ids = cluster_based_recommendations(input_song_name, num_recommendations=5)
cluster_pred
hybrid_pred, hybrid_ids = hybrid_recommendation(input_song_name, num_recommendations=5)
hybrid_pred
results_1 = []
results_1.append({'Model': 'Popularity', **evaluate_all(genre_truth_ids, pop_ids)})
results_1.append({'Model': 'KNN', **evaluate_all(genre_truth_ids, knn_ids)})
results_1.append({'Model': 'Clustering', **evaluate_all(genre_truth_ids, cluster_ids)})
results_1.append({'Model': 'Hybrid', **evaluate_all(genre_truth_ids, hybrid_ids)})
eval_df_1 = pd.DataFrame(results_1)
eval_df_1
## Out of Phase by Proem
input_song_name = 'Out of Phase'
song = data.loc[data['track_name'] == input_song_name]
song = song[0:1]
song
input_song_id = song['track_id'].iloc[0]
input_song_id
pop_pred, pop_ids = popularity_recommendation(input_song_name, num_recommendations=5)
pop_pred
knn_pred, knn_ids = knn_based_recommendations(input_song_name, num_recommendations=5)
knn_pred
cluster_pred, cluster_ids = cluster_based_recommendations(input_song_name, num_recommendations=5)
cluster_pred
hybrid_pred, hybrid_ids = hybrid_recommendation(input_song_name, num_recommendations=5)
hybrid_pred[['track_name', 'artists']]
results_2 = []
results_2.append({'Model': 'Popularity', **evaluate_all(genre_truth_ids, pop_ids)})
results_2.append({'Model': 'KNN', **evaluate_all(genre_truth_ids, knn_ids)})
results_2.append({'Model': 'Clustering', **evaluate_all(genre_truth_ids, cluster_ids)})
results_2.append({'Model': 'Hybrid', **evaluate_all(genre_truth_ids, hybrid_ids)})
eval_df_2 = pd.DataFrame(results_2)
eval_df_2
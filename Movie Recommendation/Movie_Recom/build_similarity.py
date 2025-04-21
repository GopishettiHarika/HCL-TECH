import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load your dataset
movies = pd.read_csv('dataset.csv')

# Handle missing values
movies['title'] = movies['title'].fillna('')
movies['genre'] = movies['genre'].fillna('')
movies['overview'] = movies['overview'].fillna('')

# Combine text features
movies['tags'] = movies['title'] + " " + movies['genre'] + " " + movies['overview']

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Save files
pickle.dump(movies, open("movies_list.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

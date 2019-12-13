""" Prediction of Users based on Tweet embeddings """
import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from .models import User
from .twitter import BASILICA

""" Determine and return which user is more likely to tweet something """
def predict_user(user1_name, user2_name, tweet_text, cache=None):




    """
    # returns
        true if user1 has at leasto ne tweet that is closer to the sample tweet.
        False otherwise.
    """
    user1 = User.query.filter(User.name == user1_name).one()
    user2 = User.query.filter(User.name == user2_name).one()
    user1_embeddings = np.array([tweet.embedding for tweet in user1.tweets])
    user2_embeddings = np.array([tweet.embedding for tweet in user2.tweets])
    user1_neigh = NearestNeighbors(metric='cosine').fit(user1_embeddings)
    user2_neigh = NearestNeighbors(metric='cosine').fit(user2_embeddings)
    tweet_embedding = BASILICA.embed_sentence(tweet_text, model='twitter')
    tweet_embedding = np.array(tweet_embedding).reshape(1, -1)
    user_1_neigh_dist, _ = user1_neigh.kneighbors(X=tweet_embedding, n_neighbors=1)
    user_2_neigh_dist, _ = user2_neigh.kneighbors(X=tweet_embedding, n_neighbors=1)
    user_1_neigh_dist = user_1_neigh_dist[0][0]
    user_2_neigh_dist = user_2_neigh_dist[0][0]
    #embeddings = np.vstack([user1_embeddings, user2_embeddings])
    #labels = np.concatenate([np.ones(len(user1.tweets)),
                             #np.zeros(len(user2.tweets))])
    #log_reg = LogisticRegression().fit(embeddings, labels)
    #cache and cache.set(user_set, pickle.dumps(log_reg))
    #tweet_embedding = BASILICA.embed_sentence(tweet_text, model='twitter')
    return user_1_neigh_dist > user_2_neigh_dist
    #return log_reg.predict(np.array(tweet_embedding).reshape(1, -1))

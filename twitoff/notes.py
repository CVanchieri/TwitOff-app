""" Terminal commands to run app """
# terminal command to run web app in development environment.
FLASK_APP=twitoff:APP FLASK_ENV=development flask run
# terminal command to run app shell to add data to SQL.
FLASK_APP=twitoff:APP flask shell

""" Terminal commands to create database, users, and tweets """
# imports necessary to add users & tweets.
from twitoff.models import DB, User, Tweet
# imports necessary for twitter API.
from twitoff.twitter import *
# create SQL data base.
DB.create_all()
#add users & tweets.
u1 = User(name='WilliamShakespeare')
t1 = Tweet(text='All that glitters is not gold. by WilliamShakespeare')
t2 = Tweet(text='By the pricking of my thumbs, Something wicked this way comes. Open, locks, Whoever knocks! by WilliamShakespeare')
t3 = Tweet(text='The lady doth protest too much, methinks. by WilliamShakespeare')
u2 = User(name='AlbertEinstein')
t4 = Tweet(text='Insanity: doing the same thing over and over again and expecting different results.')
t5 = Tweet(text='Imagination is more important than knowledge.')
t6 = Tweet(text='No problem can be solved from the same level of consciousness that created it.')
u3 = User(name='BenjaminFranklin')
t7 = Tweet(text='Either write something worth reading or do something worth writing.')
t8 = Tweet(text='Three may keep a secret, if two of them are dead.')
t9 = Tweet(text='Tell me and I forget, teach me and I may remember, involve me and I learn.')
# add the tweets to the user.
u1.tweets.append(t1)
u1.tweets.append(t2)
u1.tweets.append(t3)
u2.tweets.append(t4)
u2.tweets.append(t5)
u2.tweets.append(t6)
u3.tweets.append(t7)
u3.tweets.append(t8)
u3.tweets.append(t9)
# add the users to the database.
DB.session.add(u1)
DB.session.add(u2)
DB.session.add(u3)
# commit new data to the file.
DB.session.commit()

# Twitter API Information.
Consumer API keys
FLmtX40Hh9WrOgxkmDYBeSsxC (API key)
4ZO9szOPoWumTVSRK4ek8NVYBAp4IVnwEyvTg3vWCQCGuT32st (API secret key)
Access token & access token secret
56291872-NDZU2F8MzxFGK5MnlrXelBreaxSHD9xWZykZPIy2f (Access token)NEW
rmc9KkB7AhoCLYdnUOGLj3eiJjWBhOxx8cZszBe7JSwO5 (Access token secret)

from twitoff.twitter import *
DB.drop_all()
DB.create_all()

twitter_user = TWITTER.get_user('austen')
embedding = BASILICA.embed_sentence(tweet_text, model='twitter')
tweets = twitter_user.timeline(count=200, exclude_replies=True, include_rts=False, tweet_mode='extended')
db_user = User(id)

tweets[0].full_text

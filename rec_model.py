## Importing required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

## Loading the databases
books_multiple_tags = pd.read_csv('books.csv')

## Performing TF-IDF on the corpus column and using cosine_sim to get similarity values
tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books_multiple_tags['tags'].values.astype('U'))
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

titles_corpus = books_multiple_tags[['original_title', 'authors']]
indices_corpus = pd.Series(books_multiple_tags.index, index = books_multiple_tags['original_title'])

def corpus_recommendations(title):
  idx = indices_corpus[title]
  sim_scores = list(enumerate(cosine_sim_corpus[idx]))        # Creating a list of all the values in the cosine_sim vector for that specific book_id
  sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse = True)      # Sorting those values in reverse order (Most similar at the top)
  sim_scores = sim_scores[1:21]     # Selecting top 20
  book_indices = [i[0] for i in sim_scores]    
  return titles_corpus.iloc[book_indices]

model = corpus_recommendations
pickle.dump(model, open('model.pkl','wb'))
#print("Enter a title related to which you would like to see recommendations:")
#book_title = input()
#print(corpus_recommendations(book_title))
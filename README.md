# Book Recommendation System

This repository contains Python code for building a book recommendation system. The system is based on the concept of TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to suggest books that are similar to a given book title.

## Overview
The code performs the following steps:

**Importing Required Libraries:** The required libraries can be imported using the following command:
```bash
pip install -r requirements.txt
```

**Loading the Database:** It loads a database of books from `books.csv`. This database contains information about books, including their titles, authors, and tags.

**TF-IDF Vectorization and Cosine Similarity:** The code processes the tags of the books in the database using TF-IDF vectorization. This technique converts textual data into numerical vectors. Then, it calculates the cosine similarity between book tags to determine how similar they are.

**Recommendation Function:** The corpus_recommendations function takes a book title as input and returns a list of book recommendations. It does this by first finding the index of the input book title in the dataset, calculating the cosine similarity scores between the input book and all other books, sorting the scores in descending order, and selecting the top 20 most similar books as recommendations.

**Model Serialization:** The code serializes the recommendation model using Pickle, allowing it to be saved and reused in the Flask API.

## Usage
To run the API on your local server, run the following command:
```bash
python app.py
```
<img src="demo.jpg" width=50% height=1% align="centre">
![View of output](./demo.jpg)

Simply provide the title of a book related to the genre or theme you're interested in, and the system will suggest similar books.

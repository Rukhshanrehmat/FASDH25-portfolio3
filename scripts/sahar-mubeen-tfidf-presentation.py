# TF-IDF Visualisation Script

import pandas as pd
import plotly.express as px
import os

# Load the TF-IDF cosine similarity dataset
df = pd.read_csv("../data/dataframes/tfidf/tfidf-over-0.3.csv")

# Check the first few rows to understand the structure (class-safe)
print("----- First 5 Rows of the Dataset -----")
print(df.head())

# Sort the data by similarity score in descending order
top_similar = df.sort_values(by='similarity', ascending=False).head(10)

# Print the top 10 pairs and their scores (helps confirm what's being visualised)
print("\n----- Top 10 Most Similar Article Pairs -----")
print(top_similar[['filename-1', 'filename-2', 'similarity']])

# Create a new column combining filenames of both articles in the pair
top_similar['pair'] = top_similar['filename-1'] + " & " + top_similar['filename-2']

# Print new 'pair' column for confirmation
print("\n----- Article Pair Labels for the Graph -----")
print(top_similar['pair'])

# Create a bar chart using plotly express
fig = px.bar(
    top_similar, 
    x='pair', 
    y='similarity', 
    title='Top 10 Most Similar Articles Based on TF-IDF Cosine Similarity',
    labels={'pair': 'Article Pair', 'similarity': 'Cosine Similarity Score'}
)

# Display the plot
fig.show()

fig.write_html("output/similarity_tfidf.html")


# TF-IDF Exploration Script

import pandas as pd
import plotly.express as px

# Load the TF-IDF cosine similarity dataset
df = pd.read_csv("../data/dataframes/tfidf/tfidf-over-0.3.csv")

# Print information in the datase and print number of rows and columns
print("\n----- Dataset Shape (rows, columns) -----")
print(df.shape)

# Preview the first 10 rows of the dataset
print("\n----- First 10 Rows -----")
print(df.columns)
print(df.head(10))

# Show column names
print("\n----- Column Names -----")
print(df.columns)

# Show number of rows and columns
print("Number of rows and columns:")
print(df.shape)

# Show basic statistics about the similarity scores
print(df['similarity'].describe())

# Filter article pairs with similarity > 0.7
print("\n----- Articles with Similarity > 0.7 -----")
high_sim = df[df['similarity'] > 0.7]
print(high_sim)

# Check how many article pairs are very similar (over 0.9)
print("\n----- Count of Pairs with Similarity > 0.9 -----")
print(len(df[df['similarity'] > 0.9]))

# Show the lowest similarity value
print("\n----- Minimum Similarity Value in Dataset -----")
print(df['similarity'].min())

# Create a histogram of similarity scores
fig = px.histogram(
    df,
    x='similarity',
    nbins=20,  # Number of bars/bins in the histogram
    title='Distribution of TF-IDF Cosine Similarity Scores',
    labels={'similarity': 'Cosine Similarity Score'}
)

# Show the plot
fig.show()

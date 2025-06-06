# exploration.py

import pandas as pd  # Import pandas for data manipulation and analysis
import plotly.express as px  # Import Plotly Express for quick interactive visualizations
import plotly.graph_objects as go  # Import Plotly Graph Objects for advanced charting like tables
import os  # Import os for file path operations

# Define the path to the topic model CSV file
DATA_PATH = os.path.join("..", "data", "dataframes", "topic-model", "topic-model.csv")

# Load the dataset into a pandas DataFrame
df = pd.read_csv(DATA_PATH)

# Print basic information about the DataFrame including data types and non-null counts
print("\n--- Basic Info ---")
print(df.info())

# Display the first 5 rows of the dataset
print("\n--- First 5 Rows ---")
print(df.head())

# Print the names of all columns in the DataFrame
print("\n--- Columns ---")
print(df.columns)

# Check for missing values in each column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Display the count of each unique topic label
print("\n--- Unique Topics ---")
print(df["Topic"].value_counts())

# Filter out rows where the topic is -1 
df_clean = df[df["Topic"] != -1].copy()

# Combine year, month, and day columns into a single datetime column
df_clean["date"] = pd.to_datetime(df_clean[["year", "month", "day"]])

# Count the number of documents per topic
topic_counts = df_clean["Topic"].value_counts().reset_index()

# Rename the columns for clarity
topic_counts.columns = ["Topic", "Count"]

# Create a bar chart showing the number of documents per topic
fig1 = px.bar(
    topic_counts,
    x="Topic",
    y="Count",
    title="Distribution of Topics in Corpus (Excluding -1)",
    labels={"Topic": "Topic Number", "Count": "Document Count"},
    height=500
)

# Save the bar chart as an interactive HTML file
fig1.write_html("../output/explore_topic_distribution.html")

# Group data by topic and month to compute document counts per topic per month
df_monthly = df_clean.groupby(["Topic", pd.Grouper(key="date", freq="ME")]).size().reset_index(name="Count")

# Create a line chart showing topic trends over time (monthly)
fig2 = px.line(
    df_monthly,
    x="date",
    y="Count",
    color="Topic",
    title="Topic Trends Over Time (Monthly)",
    labels={"date": "Date", "Count": "Document Count"},
    height=600
)

# Save the line chart as an interactive HTML file
fig2.write_html("../output/explore_topic_trends_over_time.html")

# Aggregate the most frequent keyword in each topic for four keyword columns
top_words = (
    df_clean.groupby("Topic")[["topic_1", "topic_2", "topic_3", "topic_4"]]
    .agg(lambda x: pd.Series.mode(x)[0])  # Select the most common value in each column
    .reset_index() 
) # This section i.e top_words, was done with the help of AI as I tried adding keywords together but was unable to do so, So i took help from ChatGpt

# Create a table displaying the top 4 keywords for each topic
fig3 = go.Figure(data=[go.Table(
    header=dict(values=["Topic", "Word 1", "Word 2", "Word 3", "Word 4"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[top_words.Topic, top_words.topic_1, top_words.topic_2,
                       top_words.topic_3, top_words.topic_4],
               fill_color='lavender',
               align='left'))
])

# Set the title of the keyword table
fig3.update_layout(title="Top Keywords per Topic")

# Save the keyword table as an interactive HTML file
fig3.write_html("../output/explore_topic_keywords_table.html")

# Print confirmation of saved visualizations
print("Exploration visualizations saved:")
print(" - explore_topic_distribution.html")
print(" - explore_topic_trends_over_time.html")
print(" - explore_topic_keywords_table.html")

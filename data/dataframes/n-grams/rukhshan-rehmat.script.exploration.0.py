# Import necessary libraries
import pandas as pd
import plotly.express as px

# File path to the 2-gram CSV
file_path = r"C:\Users\LENOVO\Desktop\FASDH25-portfolio3\data\dataframes\n-grams\2-gram\2-gram-year-month.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Print the first 10 rows to inspect the data
print("First 10 rows of the data:")
print(df.head(10))

# Sort by count-sum (total count) descending to find most frequent 2-grams
top_10 = df.sort_values(by="count-sum", ascending=False).head(10)
print("\nTop 10 most frequent 2-grams by count-sum:")
print(top_10)

# Filter the DataFrame for specific 2-grams to explore their trends
selected_2grams = ["humanitarian add", "air strike", "cease fire"]
df_filtered = df[df["2-gram"].isin(selected_2grams)]

# Check the filtered data
print("\nFiltered 2-grams data:")
print(df_filtered)

# Create a 'date' column from year and month for proper time series plotting
df_filtered["date"] = pd.to_datetime(dict(year=df_filtered["year"], month=df_filtered["month"], day=1))

# Plot time series of selected 2-grams' monthly counts
fig = px.line(
    df_filtered,
    x="date",
    y="count-sum",
    color="2-gram",
    title="Monthly Frequency of Selected 2-grams",
    labels={"date": "Date", "count-sum": "Frequency", "2-gram": "2-gram Phrase"}
)
fig.write_html("outputs/time series of selected 2-grams monthly counts.html")
fig.show()

# File path
file_path = r"C:\Users\LENOVO\Desktop\FASDH25-portfolio3\data\dataframes\n-grams\2-gram\2-gram-year-month.csv"

# Load data
df = pd.read_csv(file_path)

# List of stopwords (common English stopwords)
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "s", "said", "one", "would", "also",
    "since", "two", "including"
])

# Filter data for years 2021 to 2024
df_filtered_years = df[(df["year"] >= 2021) & (df["year"] <= 2024)]

# Function to check if a 2-gram contains stop words
def contains_stopword(phrase):
    # split the 2-gram into words
    words = phrase.lower().split()
    # return True if any word in the phrase is a stopword
    return any(word in stop_words for word in words)

# Remove rows where the 2-gram contains any stop word
df_filtered = df_filtered_years[~df_filtered_years["2-gram"].apply(contains_stopword)]

# Aggregate counts by 2-gram over all months and years
top_50 = (
    df_filtered.groupby("2-gram")["count-sum"]
    .sum()
    .reset_index()
    .sort_values("count-sum", ascending=False)
    .head(50)
)

print("Top 50 2-grams (stop words removed, 2021-2024):")
print(top_50)

# Plot top 50 2-grams by total counts
fig = px.bar(
    top_50,
    x="count-sum",
    y="2-gram",
    orientation="h",
    title="Top 50 Bi-grams (Stop Words Removed, 2021-2024)",
    labels={"count-sum": "Total Frequency", "2-gram": "2-gram Phrase"}
)

fig.update_layout(yaxis={"categoryorder":"total ascending"})  # highest at top
fig.write_html("outputs/top 50 2-grams by total counts_barchart.html")
fig.show()
















import pandas as pd
import plotly.express as px
import os
import re
import webbrowser
from nltk.corpus import stopwords
from nltk import download

# Ensure NLTK stopwords are downloaded
download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update({"s", "said", "al", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
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
    "just", "don", "should", "now", "15", "s", "said", "one", "would", "told",
    "also", "since", "two", "al", "including"

})

# File path for 2-gram data
file_path = r"C:\Users\LENOVO\Desktop\FASDH25-portfolio3\data\dataframes\n-grams\2-gram\2-gram-year-month.csv"

# Sympathy and Aggression 2-grams      #learned from chat gpt
sympathy_phrases = {
    "humanitarian aid", "cease fire", "medical help", "international support", "innocent civilians"
}
aggression_phrases = {
    "air strike", "military attack", "rocket fire", "dead bodies", "civilian casualties"
}

# Helper to filter valid phrases
def is_valid_ngram(ngram):
    words = ngram.lower().split()
    return all(re.fullmatch(r"\b\w+\b", word) and word not in stop_words for word in words)

# Load and clean data
df = pd.read_csv(file_path)
df.columns = ["year", "month", "ngram", "count", "mean"]
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df["month"] = df["date"].dt.to_period("M").astype(str)
df = df[df["year"] > 2020]
df = df[df["ngram"].apply(lambda x: isinstance(x, str) and is_valid_ngram(x))]    # learned from chat gpt
df["ngram_lower"] = df["ngram"].str.lower()

# Classify as sympathy or aggression    #learned from chat gpt
df["theme"] = df["ngram_lower"].apply(
    lambda x: "Sympathy" if x in sympathy_phrases else ("Aggression" if x in aggression_phrases else None)
)
df = df[df["theme"].notna()]

# Calculate relative count per month
monthly_totals = df.groupby("month")["count"].sum().rename("monthly_word_total").reset_index()
df = df.merge(monthly_totals, on="month", how="left")
df["relative_count"] = df["count"] / df["monthly_word_total"]

# Plot
fig = px.bar(
    df,
    x="month",
    y="relative_count",
    color="ngram",
    facet_row="theme",
    title="2-Gram Sympathy vs Aggression",
    labels={"month": "Month", "relative_count": "Relative Frequency", "ngram": "Phrase"},
    barmode="stack",
    height=1200
)

# Save and open
output_path = os.path.abspath("outputs/2gram_sympathy_aggression.html")
os.makedirs("outputs", exist_ok=True)
fig.write_html(output_path)
webbrowser.open(f"file://{output_path}")


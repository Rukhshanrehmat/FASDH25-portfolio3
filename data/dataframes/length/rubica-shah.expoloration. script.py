# === Imports ===
import os
import pandas as pd
import plotly.express as px

# === Load datasets ===
df_len = pd.read_csv("length-year-month.csv")
df_topics = pd.read_csv("../topic-model/topic-model.csv")
# === Create output folder ===
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# === Convert to datetime ===
df_len["date"] = pd.to_datetime(df_len[["year", "month"]].assign(day=1))
df_topics["date"] = pd.to_datetime(df_topics[["year", "month"]].assign(day=1))

# === Merge both datasets ===
df = pd.merge(df_len, df_topics, on="date", how="inner")
df = df.sort_values("date")
# === Define stopwords ===
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
    "since", "two", "including", "reuters", "mr", "mrs", "news", "say", "says"
])

# === Function to remove stopwords ===
def remove_stopwords(text, stop_words):
    if pd.isnull(text):
        return ""
    words = text.lower().split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)

# === Clean article text if available ===
if "raw_text" in df.columns:
    df["clean_text"] = df["raw_text"].apply(lambda x: remove_stopwords(x, stop_words)) # learn from chatgpt

# === Topic analysis setup ===
topic_columns = [col for col in df.columns if col.lower().startswith("topic")]
df[topic_columns] = df[topic_columns].apply(pd.to_numeric, errors='coerce')

# === Print useful information ===
print("=== Combined Dataset Preview ===")
print(df.head())

print("\n=== Topic Columns Data Types ===")
print(df[topic_columns].dtypes)

print("\n=== Average Topic Proportions ===")
print(df[topic_columns].mean().sort_values(ascending=False))

# === Plot 1: Line plot of length and top 2 topic trends ===
fig = px.line(
    df,
    x="date",
    y=["length-mean"] + topic_columns[:2],
    title="Average Article Length and Top 2 Topic Proportions Over Time",
    labels={"value": "Value", "variable": "Metric"},
    markers=True
)
fig.write_html(os.path.join(output_folder,"lineplot_length_topic.html"))
fig.write_image(os.path.join(output_folder,"lineplot_length_topic.png"))
fig.show()

# === Plot 2: Scatter plot of topic proportion vs article length ===
for topic in topic_columns[:2]:
    fig = px.scatter(
        df,
        x=topic,
        y="length-mean",
        title=f"Average Article Length vs {topic}",
        labels={topic: f"{topic} Proportion", "length-mean": "Average Length"},
        trendline="ols"
    )
    fig.write_html(os.path.join(output_folder,f"scatter_{topic}_vs_length.html"))
    fig.write_image(os.path.join(output_folder,f"scatter_{topic}_vs_length.png"))
    fig.show()

# === Plot 3: Number of months each topic was dominant ===
df["top_topic"] = df[topic_columns].idxmax(axis=1)
top_topic_counts = df["top_topic"].value_counts()
print("\n=== Top Topic Counts ===")
print(top_topic_counts)

fig = px.bar(
    top_topic_counts,
    title="Number of Months Each Topic Was Dominant",
    labels={"value": "Count", "index": "Topic"}
)
fig.write_html(os.path.join(output_folder,"bar_dominant_topics.html"))
fig.write_image(os.path.join(output_folder,"bar_dominant_topics.png"))
fig.show()

# === Plot 4: Topic trends over time ===
fig = px.line(
    df,
    x="date",
    y=topic_columns[:3],
    title="Top 3 Topic Trends Over Time",
    labels={"value": "Proportion", "variable": "Topic"},
    markers=True
)
fig.write_html(os.path.join(output_folder,"lineplot_top3_topics.html"))
fig.write_image(os.path.join(output_folder,"lineplot_top3_topics.png"))
fig.show()

# === Plot 5: Average length by topic proportion levels (first topic) ===
first_topic = topic_columns[0]
df[f"{first_topic}_level"] = pd.qcut(df[first_topic], 3, labels=["Low", "Medium", "High"])
grouped = df.groupby(f"{first_topic}_level")["length-mean"].mean()
print(f"\n=== Avg Article Length by {first_topic} Level ===")
print(grouped)

# === Plot 6: Boxplot of article length grouped by top topic ===
fig = px.box(
    df,
    x="top_topic",
    y="length-mean",
    title="Article Length by Top Topic"
)
fig.write_html(os.path.join(output_folder,"boxplot_length_by_topic.html"))
fig.write_image(os.path.join(output_folder,"boxplot_length_by_topic.png"))
fig.show()

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

# === Convert to datetime and merge ===
df_len["date"] = pd.to_datetime(df_len[["year", "month"]].assign(day=1))
df_topics["date"] = pd.to_datetime(df_topics[["year", "month"]].assign(day=1))
df = pd.merge(df_len, df_topics, on="date", how="inner").sort_values("date")

# === Identify topic columns ===
topic_columns = [col for col in df.columns if col.lower().startswith("topic")]
df[topic_columns] = df[topic_columns].apply(pd.to_numeric, errors='coerce')

# === Plot 1: Article length and top 2 topic trends ===
fig = px.line(
    df,
    x="date",
    y=["length-mean", topic_columns[0], topic_columns[1]],
    title="Article Length and Topic Proportions Over Time",
    labels={"value": "Value", "variable": "Metric"},
    markers=True
)
fig.write_html(os.path.join(output_folder, "presentation_lineplot.html"))
fig.write_image(os.path.join(output_folder, "presentation_lineplot.png"))
fig.show()

# === Plot 2: Length vs Topic Proportion (Scatter with Trendline) ===
for topic in topic_columns[:2]:
    fig = px.scatter(
        df,
        x=topic,
        y="length-mean",
        title=f"Average Article Length vs {topic}",
        trendline="ols",
        labels={topic: f"{topic} Proportion", "length-mean": "Article Length"}
    )
    fig.write_html(os.path.join(output_folder, f"presentation_scatter_{topic}.html"))
    fig.write_image(os.path.join(output_folder, f"presentation_scatter_{topic}.png"))
    fig.show()

# === Plot 3: Topic Dominance Over Time ===
df["top_topic"] = df[topic_columns].idxmax(axis=1)
top_topic_counts = df["top_topic"].value_counts()

fig = px.bar(
    top_topic_counts,
    title="Months Each Topic Was Dominant",
    labels={"value": "Number of Months", "index": "Topic"}
)
fig.write_html(os.path.join(output_folder, f"presentation_scatter_{topic}.html"))
fig.write_image(os.path.join(output_folder, "presentation_bar_top_topic.png"))
fig.show()

import pandas as pd  # Imports pandas library for data manipulation
import plotly.express as px  # Imports Plotly Express for quick and easy plotting
import os  # Imports os module for file and path operations

base_dir = os.path.join("..")  # Sets base directory to the parent folder
topic_model_path = os.path.join(base_dir, "data", "dataframes", "topic-model", "topic-model.csv")  # Constructs path to the topic model CSV file
output_dir = os.path.join(base_dir, "output")  # Sets output directory path
os.makedirs(output_dir, exist_ok=True)  # Creates the output directory if it doesn't already exist

df = pd.read_csv(topic_model_path)  # Loads the topic model CSV into a DataFrame

df = df[df['Topic'] != -1]  # Removes rows with Topic -1, which typically represent uncategorized or noise topics

df['Topic_Keywords'] = df[['topic_1', 'topic_2', 'topic_3', 'topic_4']].agg(', '.join, axis=1)  # Combines the top 4 topic keywords into a single string

pronouns = {"my", "he", "she", "we", "they", "him", "her", "us", "you", "i"}  # Defines a set of pronouns to filter out pronoun-heavy topics
def is_pronoun_topic(keywords):  # Defines function to check if a topic is dominated by pronouns
    kws = [kw.strip().lower() for kw in keywords.split(',')]  # Splits keyword string and normalizes to lowercase
    return sum(kw in pronouns for kw in kws) >= 3  # Returns True if 3 or more keywords are pronouns

df = df[~df['Topic_Keywords'].apply(is_pronoun_topic)]  # Filters out pronoun-heavy topics using the custom function

topic_counts = df.groupby(['Topic', 'Topic_Keywords']).size().reset_index(name='Article_Count')  # Counts number of articles per topic and keyword group
topic_counts = topic_counts.sort_values(by='Article_Count', ascending=False)  # Sorts topics by article count in descending order
total_articles = topic_counts['Article_Count'].sum()  # Calculates total number of articles after filtering
topic_counts['Percentage'] = (topic_counts['Article_Count'] / total_articles * 100).round(2)  # Computes percentage of total articles for each topic

top_n = 10  # Defines the number of top topics to display
top_topics = topic_counts.head(top_n).copy()  # Selects the top N topics based on article count
top_topics['Label'] = top_topics['Topic_Keywords'] + ' (' + top_topics['Percentage'].astype(str) + '%)'  # Creates a label combining keywords and percentage

print(f"Top {top_n} topics (excluding pronoun-heavy ones):")  # Prints a header for the topic summary
for idx, row in top_topics.iterrows():  # Loops through each top topic
    print(f"- Topic {row['Topic']}: {row['Topic_Keywords']} ({row['Article_Count']} articles, {row['Percentage']}%)")  # Prints topic details

fig_bar = px.bar(  # Creates a horizontal bar chart of the top topics
    top_topics,
    x='Article_Count',
    y='Label',
    orientation='h',
    title=f"Top {top_n} Topics in the Corpus\n(Media Focus and Thematic Saturation)",
    labels={'Article_Count': 'Number of Articles', 'Label': 'Topic (Top 4 Keywords)'},
    height=600
)
fig_bar.update_layout(yaxis=dict(autorange="reversed"))  # Reverses y-axis to show the highest topic on top
fig_bar.update_traces(marker_color='crimson')  # Sets the bar color to crimson

bar_html_path = os.path.join(output_dir, "presentation_topic_counts.html")  # Constructs path to save the bar chart HTML
fig_bar.write_html(bar_html_path)  # Saves the bar chart as an interactive HTML file

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])  # Converts year, month, day columns to a single datetime column

df_top = df[df['Topic'].isin(top_topics['Topic'])].copy()  # Filters the original DataFrame to include only top N topics

top_5_topics = top_topics.head(5)['Topic'].tolist()  # Selects the top 5 topics for trend analysis
df_trend = df[df['Topic'].isin(top_5_topics)].copy()  # Creates a DataFrame with only the top 5 topics

df_trend['month_start'] = df_trend['date'].dt.to_period('M').dt.to_timestamp()  # Extracts the first day of each month from date

monthly_trend = df_trend.groupby(['month_start', 'Topic', 'Topic_Keywords']).size().reset_index(name='Article_Count')  # Groups by month, topic, and keywords to count articles

monthly_trend['Rolling_Count'] = monthly_trend.groupby('Topic')['Article_Count'].transform(  # Applies rolling average to smooth trends
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

fig_line = px.line(  # Creates a line chart to visualize topic trends over time
    monthly_trend,
    x='month_start',
    y='Rolling_Count',
    color='Topic_Keywords',
    markers=True,
    title="Top 5 Topic Trends in the Corpus Over Time (3-Month Rolling Average)",
    labels={
        'month_start': 'Month',
        'Rolling_Count': 'Avg Article Count',
        'Topic_Keywords': 'Topic (Top 4 Keywords)'
    },
    height=600
)

fig_line.update_layout(  # Updates layout settings for the line chart
    yaxis=dict(title='Average Article Count'),
    xaxis=dict(title='Month'),
    legend_title='Topic (Top 4 Keywords)',
    template='plotly_white'
)

line_html_path = os.path.join(output_dir, "presentation_topic_trends.html")  # Constructs path to save the line chart HTML
fig_line.write_html(line_html_path)  # Saves the line chart as an interactive HTML file

print("\nInterpretation:")  # Prints interpretation note
print(f"- The bar chart shows the top {top_n} topics by article count, highlighting media focus and thematic saturation.")  # Explains bar chart
print("- The line chart shows rolling 3-month average trends for the top 5 topics, making long-term changes more visible.")  # Explains line chart
print(f"\nCharts saved to:\n - {bar_html_path}\n - {line_html_path}")  # Prints file paths for saved charts

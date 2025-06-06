# Import necessary libraries
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords

# Download stopwords (only needs to run once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# File path for the 1-gram monthly dataset
file_path = r"C:\Users\LENOVO\Desktop\FASDH25-portfolio3\data\dataframes\n-grams\1-gram.csv"



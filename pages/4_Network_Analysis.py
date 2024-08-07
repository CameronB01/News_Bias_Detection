import streamlit as st
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import umap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.cluster import KMeans
from transformers import pipeline
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Network Analysis", page_icon=":bar_chart:", layout="wide")

main_list = pd.read_csv("articles_with_all_test.csv")

main_list = main_list.dropna(subset=["text"])
main_list = main_list.drop_duplicates(subset=["title", "authors", "top_image"])
main_list = main_list.drop_duplicates().reset_index()


entities = pd.read_csv('entities.csv')


st.title("Network Analysis:")


st.markdown("""
This page provides an interactive exploration of article sentiments and emotions across various topics and biases. Utilize visualizations and interactive controls to delve into the sentiment distributions, popular locations, and emotional landscapes of your dataset.

### Key Features:

1. **Distribution of Article Sentiment:**
   - A histogram that visualizes the sentiment distribution of articles, enriched with a marginal violin plot. This offers a comprehensive view of how articles are spread across different sentiment scores.
   - The sentiment distribution plot shows the emotional tone of our news articles. It helps analyze changes in public sentiment over time. By showing the prevalence of different sentiments, this chart shows how emerging trends are perceived in the media and reveals the feelings toward a given topic that each news network portrays.
   - The sentiment score ranges from -1 to 1, where -1 represents a negative sentiment, 0 is neutral, and 1 is positive.
   - Because we have only been able to scrape news articles for about 2 months, this data is not very rich. However, if the scraping continues, it would be fascinating to see how sentiment changes over time and the sentiment towards different topics.
""")



bias_mapping = {0: 'Center', 1: 'Left', 2: 'Right'}

entities['bias'] = entities['bias'].map(bias_mapping)

label_mapping = {'PERSON': 'PERSON - People, including fictional', 
                 'NORP': 'NORP - Nationalities or religious or political groups', 
                 'FAC': 'FACILITY - Buildings, airports, highways, bridges, etc.', 
                 'ORG': 'ORGANIZATION - Companies, agencies, institutions, etc.',
                 'GPE': 'GPE - Countries, cities, states',
                 'LOC': 'LOCATION - Non-GPE locations, mountain ranges, bodies of water',
                 'PRODUCT': 'PRODUCT - Vehicles, weapons, foods, etc. (Not services)',
                 'EVENT': 'EVENT - Named hurricanes, battles, wars, sports events, etc.',
                 'WORK_OF_ART': 'WORK OF ART - Titles of books, songs, etc.',
                 'LAW': 'LAW - Named documents made into laws',
                 'LANGUAGE': 'LANGUAGE - Any named language',
                 'DATE': 'DATE - Absolute or relative dates or periods',
                 'TIME': 'TIME - Times smaller than a day',
                 'PERCENT': 'PERCENT - Percentage, including "%" ',
                 'MONEY': 'MONEY - Monetary values, including unit',
                 'QUANTITY': 'QUANTITY - Measurements, as of weight or distance',
                 'ORDINAL': 'ORDINAL - "first", "second", etc.',
                 'CARDINAL': 'CARDINAL - Numerals that do not fall under another type'}

entities['label'] = entities['label'].map(label_mapping)


# Create the Plotly histogram with KDE
fig = px.histogram(
    main_list,
    x='sentiment_score',
    nbins=30,  # Number of bins
    title=' ',
    labels={'sentiment_score': 'Article Sentiment', 'count': 'Frequency'},
    opacity=0.7,  # Opacity of the bars
    marginal='violin',  # Add a violin plot for additional context (optional)
)

# Customize layout
fig.update_layout(
    xaxis_title='Article Sentiment',
    yaxis_title='Density',
    height=600,  # Set the height of the figure
    width=1200,  # Set the width of the figure
    bargap=0.1,  # Gap between bars
    title_x=0.5,  # Center the title
)

# Display the Plotly chart in Streamlit
# st.title('Distribution of Article Sentiment')
st.plotly_chart(fig, use_container_width=True)



col1, col2 = st.columns(2)

with col1:


    st.markdown("""
    2. **Label Mentions by Topic and Bias:**
       - An interactive bar chart that displays the most frequently mentioned labels within selected topics and political biases. This feature helps identify specific hotspots associated with certain content and perspectives.
    """)



    label = st.selectbox(
        "Select a topic to explore:",
        (entities['label'].unique()),
    )

    political_bias = st.selectbox(
        "Select a political bias:",
        (entities['bias'].unique()),
    )

    GPE = pd.DataFrame(entities[entities['label'] == label])
    GPE['text'] = GPE['text'].str.replace(r'[^\w\s]', '', regex=True)
    GPE_counts = GPE[['text', 'bias']].value_counts().reset_index(name='count')
    GPE_counts_bias = GPE_counts[GPE_counts['bias'] == political_bias].reset_index(drop=True).head(10)


    # Create the Plotly bar chart
    fig = px.bar(
        GPE_counts_bias,
        x='count',
        y='text',
        orientation='h',
        title=' ',
        labels={'count': 'Count', 'text': 'Location'},
        width=1200,  # Width of the figure
        height=800,  # Height of the figure
        color='count',  # Use count for color differentiation
        color_continuous_scale=px.colors.sequential.Blues  # Use a color scale for the bars
    )

    # Customize layout
    fig.update_layout(
        xaxis_title='Count',
        yaxis_title='Location',
        yaxis=dict(categoryorder='total ascending'),  # Order locations by count
        title_x=0.5,  # Center the title
        coloraxis_showscale=False  # Optionally hide the color scale
    )

    # Display the Plotly chart in Streamlit
    # st.title('Top 20 Locations Mentioned (Right)')
    st.plotly_chart(fig, use_container_width=True)









with col2:
    
    
    
    
    st.markdown("""    
    3. **Emotion Analysis by Topic and Site:**
       - A radar chart that visualizes the emotional content of articles, providing insights into the predominant emotions expressed within articles on chosen topics and sites. The analysis leverages sentiment analysis to categorize emotions such as joy, anger, fear, and more.
       """)
    
    

    topic = st.selectbox(
        "Select a topic to explore:",
        (main_list['dominant_topic_text_short'].unique()),
    )

    site = st.selectbox(
        "Select a News Network:",
        (main_list['site_name'].unique()),
    )

    # Load the sentiment analysis model
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

    # Function to process a single article text
    def process_text(text):
        # Split the text into smaller chunks
        chunk_size = 500  # Adjust the chunk size as needed
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Perform sentiment analysis on each chunk
        results = []
        for chunk in chunks:
            chunk_results = classifier(chunk)
            results.extend(chunk_results)
        
        # Count the occurrences of each emotion
        all_emotions = ['joy', 'anger', 'fear', 'surprise', 'sadness', 'disgust']
        emotion_counts = Counter({emotion: 0 for emotion in all_emotions})
        emotion_counts.update([result['label'] for result in results])
        
        return emotion_counts

    # Initialize an empty Counter to aggregate results
    total_emotion_counts = Counter({emotion: 0 for emotion in ['joy', 'anger', 'fear', 'surprise', 'sadness', 'disgust']})


    filtered_articles = main_list.loc[
        (main_list['site_name'] == site) & 
        (main_list['dominant_topic_text_short'] == topic), 
        'text'
    ]


    # Iterate over the desired rows (for example, the first 10 rows)
    for text in tqdm(filtered_articles, desc="Processing Articles", unit="article"):
        emotion_counts = process_text(text)
        total_emotion_counts.update(emotion_counts)

    # Prepare the data for the DataFrame
    data = {
        'Emotion': list(total_emotion_counts.keys()),
        'Count': list(total_emotion_counts.values())
    }

    # Create the DataFrame
    df_emotions = pd.DataFrame(data)

    # Remove the last row if needed (similar to df_emotions = df_emotions[:-1])
    df_emotions = df_emotions[:-1]

    # Plot radar chart for emotion analysis using Plotly
    fig = px.line_polar(
        df_emotions,
        r='Count',
        theta='Emotion',
        line_close=True,
        template='plotly_dark',
        # title='Emotion Analysis Radar Chart',
        color_discrete_sequence=['#ff4b4b']  # Set the color for the line
    )

    # Customize the chart appearance
    fig.update_traces(fill='toself', fillcolor='rgba(255, 126, 126, 0.25)', line=dict(width=2))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),  # Hide radial axis labels
            angularaxis=dict(
                tickfont=dict(size=15, color='white'),  # Font size and color for tick labels
            ),
            bgcolor='#262730'  # Set background color of the polar chart
        ),
        plot_bgcolor='#0e1118',  # Set background color of the plot
        paper_bgcolor='#0e1118',  # Set background color of the figure
    )

    # Center the plot using columns
    col1, col2, col3 = st.columns([1, 3, 1])  # Adjust column widths
    with col2:
        st.plotly_chart(fig, use_container_width=True)
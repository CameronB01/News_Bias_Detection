import streamlit as st
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import umap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.cluster import KMeans
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Content Analysis", page_icon=":bar_chart:", layout="wide")

main_list = pd.read_csv("articles_with_all_test.csv")

main_list = main_list.dropna(subset=["text"])
main_list = main_list.drop_duplicates(subset=["title", "authors", "top_image"])
main_list = main_list.drop_duplicates().reset_index()


st.title("Content Analysis:")


st.markdown("""
This page provides an in-depth analysis of the distribution of article lengths, entity word clouds, and various metrics across different sites and topics. Through interactive visualizations, you can explore the characteristics and patterns within the dataset.

### Key Visualizations:

1. **Distribution of Article Lengths:**
   - The histogram shown below visualizes the distribution of article lengths in the dataset, measured in words. The x-axis represents article length in words, while the y-axis shows how often articles fall into each length range. The plot reveals that most articles are short, with a peak around 500 words. The distribution is right-skewed, meaning most articles are short, but there are a few that have up to around 17,500 words.
   - Understanding article length distribution is essential for analyzing news reporting and consumption. Shorter articles may focus on brief updates and breaking news, while longer ones provide in-depth analysis and comprehensive coverage of complex issues. This length variation affects topic coverage and perception, influencing trends and public sentiment. By examining article lengths, we can understand the balance between quick news bites and detailed reporting, and its effect on public opinion and regional news coverage.
   - Shorter news bites are preferred over longer articles. This makes sense, as people who click on news links from social media stay on a story for an average of 111 seconds (Mitchell et al., 2016). This causes news networks to shorten their news to keep their audience engaged. 
   - Extreme outliers are filtered out to provide a clearer view of the central tendency and spread of article lengths.
""")


# Create the Plotly histogram
fig = px.histogram(
    main_list[main_list['word_count'] < 10000],  # Filter out extreme outliers
    x='word_count',
    nbins=100,
    title=' ',
    labels={'article_length': 'Article Length (words)', 'count': 'Frequency'},
    opacity=0.75,
    marginal='violin'  # Add a marginal rug plot for additional context
)

# Update layout for better visualization
fig.update_layout(
    xaxis_title='Article Length (words)',
    yaxis_title='Frequency',
    bargap=0.1,  # Gap between bars
    title_x=0.5,  # Center the title
    height=800,  # Set the height of the figure
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)






st.markdown("""
2. **Word Cloud of All Articles:**
   - This graphic highlights the most frequent terms in the news articles dataset. Larger words indicate higher frequency, giving a quick overview of the main topics and entities covered, such as ‘Joe Biden,’ ‘new,’ ‘year,’ ‘New York,’ ‘Trump,’ and ‘Supreme Court.’
   - This visualization identifies the primary focus areas and key entities in the news articles, highlighting significant political figures, institutions, and locations. This visual really helps us get a good foundational understanding of the main themes shaping the current public discourse.
   - Understanding these main topics is very important. It provides us with a better understanding of the themes driving public sentiment and how they change over time. Joe Biden being mentioned the most makes sense as we are in the middle of an election year. From this visual, we can clearly see that the upcoming election is the most important news coverage. However, it would be interesting to see how this cloud would change in two years when all the election news has dissipated.
""")





# Combine all entity text into one large string
all_text = ' '.join(str(entity) for entity in main_list['entities'] if isinstance(entity, str))

# Generate the word cloud
wordcloud = WordCloud(width = 17000, height=5000, background_color='#0e1117', colormap='viridis').generate(all_text)

# Convert the word cloud into an image array
wordcloud_array = wordcloud.to_array()

# Create a Plotly image figure
fig = px.imshow(
    wordcloud_array,
    binary_string=True,  # Convert to binary image
    labels={'x': 'X', 'y': 'Y'},
    # title='Word Cloud of All Articles'
)

# Update layout to remove axes
fig.update_layout(
    xaxis=dict(showticklabels=False, visible=False),
    yaxis=dict(showticklabels=False, visible=False),
    margin=dict(l=0, r=0, t=30, b=0),  # Adjust margins to fit the title nicely
)
st.plotly_chart(fig, use_container_width=True)

















st.markdown("""
3. **Distribution of Articles by Topic:**
   - This bar chart shows the distribution of articles by their main topics within the dataset. Each bar represents a specific topic, with the height showing the number of articles on that topic. Users can quickly see the most covered topics, revealing key areas of interest. This helps users understand trends, prioritize reading, or guide further analysis. The chart also allows for comparing the prominence of different topics, highlighting potential gaps or underrepresented areas in the dataset.
""")




topic_sentiment_counts = main_list.groupby(['dominant_topic_text_short', 'sentiment']).size().reset_index(name='count')

# Step 2: Pivot the DataFrame to get sentiment counts for each topic
pivot_df = topic_sentiment_counts.pivot(index='dominant_topic_text_short', columns='sentiment', values='count').fillna(0)

# Step 3: Sort the DataFrame by the total count across all sentiments
pivot_df['total'] = pivot_df.sum(axis=1)
pivot_df = pivot_df.sort_values('total', ascending=True).drop(columns='total')

custom_colors = {
    'Positive': '#83c9ff',  # Blue
    'Negative': '#ffabab',  # Red
    'Neutral': '#0068c9'    # Green
}

# Step 4: Create the Plotly stacked bar chart
fig = px.bar(
    pivot_df,
    x=pivot_df.columns,
    y=pivot_df.index,
    orientation='h',
    title=' ',
    labels={'value': 'Number of Articles', 'index': 'Topic'},
    height=900,
    color_discrete_map=custom_colors
)

# Customize layout
fig.update_layout(
    barmode='stack',
    xaxis_title='Number of Articles',
    yaxis_title='Topic',
    yaxis=dict(categoryorder='total ascending'),  # Order topics by count
    title_x=0.5,  # Center the title
    showlegend=True, 
    legend_title_text='Sentiment',  # Custom legend title
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)









st.markdown("""   
4. **Box Plot of Word Counts by Site:**
   - This box plot shows the distribution of article word counts across various news sites in the dataset. Each box represents a news site and the range of article lengths it publishes. The central line in each box shows the median word count, while the box spans the interquartile range (IQR), covering the middle 50% of the data. The whiskers extend to the minimum and maximum word counts, excluding outliers, which are shown as individual points beyond the whiskers.
   - By analyzing this plot, users can see the variability and typical length of content for each site. It shows differences in content length among sites, indicating which ones publish longer or shorter articles. The plot also identifies potential outliers, pointing out unusual content lengths that may need further investigation. This analysis helps users evaluate writing styles, editorial standards, or focus areas of different news outlets.
   """)



# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = main_list['word_count'].quantile(0.25)
Q3 = main_list['word_count'].quantile(0.75)
IQR = Q3 - Q1  # Interquartile range

# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data to remove outliers
filtered_data = main_list[(main_list['word_count'] >= lower_bound) & (main_list['word_count'] <= upper_bound)]


# Create the Plotly box plot
fig = px.box(
    filtered_data,
    x='site_name',
    y='word_count',
    color='site_name',  # Color by site name for visual distinction
    title=' ',
    labels={'site_name': 'Site', 'word_count': 'Word Count'},
    width=1200,  # Width of the figure
    height=1000,  # Height of the figure
    color_discrete_sequence=px.colors.qualitative.Pastel  # Use a pastel color palette
)

# Customize layout
fig.update_layout(
    xaxis_title='Site',
    yaxis_title='Word Count',
    xaxis=dict(tickangle=45),  # Rotate x-axis labels
    showlegend=False,  # Optionally hide legend if not needed
    title_x=0.5,  # Center the title
)

# Display the Plotly chart in Streamlit
# st.title('Box Plot of Word Counts by Site')
st.plotly_chart(fig, use_container_width=True)
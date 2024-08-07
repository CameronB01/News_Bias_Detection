import streamlit as st
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import umap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Our Dataset", page_icon=":bar_chart:", layout="wide")

main_list = pd.read_csv("articles_with_all_test.csv")

main_list = main_list.dropna(subset=["text"])
main_list = main_list.drop_duplicates(subset=["title", "authors", "top_image"])
main_list = main_list.drop_duplicates().reset_index()

st.title("Our Dataset:")

st.markdown("""
This page presents a quick look at the infromation currently in our dataset as well as an breif analysis of news articles based on bias and network distribution. Through interactive visualizations, users can explore the landscape of news media and gain insights into how different networks and articles align in terms of political bias and sentiment.

### Key Visualizations:

1. **Bias Distribution:**
   - The bar chart below shows the distribution of political biases in the news articles. The categories include ‘Center,’ ‘Left,’ and ‘Right’ biases. The chart indicates that most articles have a ‘Left’ bias, with almost 30,000 articles. Articles with a ‘Center’ bias and a ‘Right’ bias are fewer, each around 15,000 articles. This plot helps us understand the political bias imbalance in our dataset. Recognizing this imbalance is crucial for interpreting the topic modeling results properly, as it highlights the need to account for this bias.
""")






# # Create a Streamlit app
# st.title('Bias Distribution')

# # Create the Seaborn count plot
# fig, ax = plt.subplots()

# # Plot with Seaborn
# sns.countplot(x=main_list['bias'], ax=ax)

# # Customize the plot
# ax.set_xlabel('Bias')
# ax.set_ylabel('Count')
# ax.set_title('Bias Distribution')

# # Customize x-tick labels
# ax.set_xticklabels(['Center', 'Left', 'Right'])

# # Display the plot in Streamlit
# st.pyplot(fig)



mapping = {0: 'Center', 1: 'Left', 2: 'Right'}

main_list['bias'] = main_list['bias'].map(mapping)

# Count the occurrences of each bias category
bias_count = main_list['bias'].value_counts().reset_index()
bias_count.columns = ['bias', 'count']

# Create a horizontal bar chart using Plotly Express
fig = px.bar(
    bias_count,
    x='count',
    y='bias',
    orientation='h',
    # title='Bias Distribution',
    labels={'bias': 'Bias', 'count': 'Count'},  # Custom labels for axes
)

# Customize the x-tick labels if needed
fig.update_yaxes(categoryorder='total ascending')  # Order by count
fig.update_xaxes(title_text='Count')
fig.update_yaxes(title_text='Bias')

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, key="bias", use_container_width=True)

















st.markdown("""
2. **News Network Distribution:**
   - The horizontal bar chart below shows the distribution of articles across different news networks. CBS News and Fox News lead with nearly 12,000 and almost 10,000 articles respectively. The Guardian and CNN Politics also contribute significantly but with fewer articles. And while it may seem that smaller networks like Bloomberg and BBC News contribute very few articles, you have to take into account that we have only been scraping this data for 2 months.
   - The plot shows that CBS News, Fox News, and The Guardian dominate other networks, shaping the narrative around trends and events. Publishing more articles increases the chance that one gets read. However, more articles are not always better. More articles mean more might get read, but the quality could decrease as more need to be produced daily.
   """)





# # Create a Streamlit app
# st.title('News Network Distribution')

# # Determine the order for the y-axis based on value counts
# order = main_list['site_name'].value_counts().index

# # Set Seaborn theme
# sns.set_theme()

# # Create a figure for the plot
# fig, ax = plt.subplots(figsize=(10, 10))

# # Create the Seaborn count plot
# sns.countplot(y=main_list['site_name'], order=order, ax=ax)

# # Customize the plot
# ax.set_xlabel('Count')
# ax.set_ylabel('Site Name')
# ax.set_title('News Network Distribution')

# # Display the plot in Streamlit
# st.pyplot(fig)





site_count = main_list['site_name'].value_counts().reset_index()
site_count.columns = ['site_name', 'count']

# Create a horizontal bar chart using Plotly Express
fig = px.bar(
    site_count,
    x='count',
    y='site_name',
    orientation='h',
    # title='News Network Distribution',
    labels={'site_name': 'Site Name', 'count': 'Count'},
    height=800,  # Set the height of the plot (in pixels)
)

# Customize the y-axis to order based on counts
fig.update_yaxes(
    categoryorder='total ascending',  # Order categories by count
    title_text='Site Name'            # Y-axis title
)

fig.update_xaxes(title_text='Count')  # X-axis title

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, key="site_name_distribution", use_container_width=True)






















st.markdown("""
3. **Sankey Diagram of News Flow:**
   - The Sankey diagram below shows the flow of news articles from various networks to their biases and sentiments. Each node represents a category (news network, bias, or sentiment), and the connecting lines (flows) show the distribution and relationships between these categories. The width of the flows represents the number of articles, providing a clear view of how news coverage is distributed across different biases and sentiments.
   - The diagram shows the relationship between news sources, political biases, and sentiments. It illustrates how articles from major networks like CBS News, Fox News, and The Guardian align with specific biases (center, left, right) and convey different sentiments (positive, negative, neutral). This visualization highlights the impact of media sources on public sentiment and potential biases in news coverage.
   - Each political bias reports about 50 percent of their articles with a positive sentiment, a few with a neutral tone, and the rest as negative. This shows how these news networks struggle to be nonpartisan on any topic. The messages are always charged in a positive or negative way.
""")




# mapping = {0: 'Center', 1: 'Left', 2: 'Right'}

# main_list['bias'] = main_list['bias'].map(mapping)

# Count occurrences for each transition
site_to_bias = main_list.groupby(['site_name', 'bias']).size().reset_index(name='count')
bias_to_party = main_list.groupby(['bias', 'sentiment']).size().reset_index(name='count')
#party_to_sentiment = df.groupby(['party', 'sentiment']).size().reset_index(name='count')

# Prepare data for Sankey diagram
source_target_data = pd.concat([
    site_to_bias.rename(columns={'site_name': 'source', 'bias': 'target', 'count': 'value'}),
    bias_to_party.rename(columns={'bias': 'source', 'sentiment': 'target', 'count': 'value'})
    #party_to_sentiment.rename(columns={'party': 'source', 'sentiment': 'target', 'count': 'value'})
])

# Reset index for consistency
source_target_data.reset_index(drop=True, inplace=True)

# Get unique labels
labels = pd.concat([source_target_data['source'], source_target_data['target']]).unique()
label_indices = {label: index for index, label in enumerate(labels)}

# Map source and target to indices
source_target_data['source'] = source_target_data['source'].map(label_indices)
source_target_data['target'] = source_target_data['target'].map(label_indices)

source_target_data.head()

import plotly.graph_objects as go

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(label_indices.keys())
    ),
    link=dict(
        source=source_target_data['source'],
        target=source_target_data['target'],
        value=source_target_data['value']
    )
)])

fig.update_layout(font_size=14, height=1000)

# Create a Streamlit app
# st.title('News Articles Sankey Diagram')

# Display the Sankey diagram in Streamlit
st.plotly_chart(fig) #, use_container_width=True)

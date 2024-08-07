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
from sklearn.decomposition import NMF


nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Article Mapping", page_icon=":bar_chart:", layout="wide")

main_list = pd.read_csv("articles_with_all_test.csv")



st.title("Article Mapping:")



st.markdown("""
This page provides an interactive visualization of document clustering and topic analysis using advanced dimensionality reduction and topic modeling techniques. Explore the thematic structure of your dataset through a 3D scatter plot, highlighting document clusters based on dominant topics.

### Key Features:

1. **Topic Modeling with NMF:**
   - Non-negative Matrix Factorization (NMF) is applied to extract topics from the text data. This technique identifies the underlying themes by decomposing the documents into a set of topics, each represented by a collection of words.

2. **Dimensionality Reduction with UMAP:**
   - Uniform Manifold Approximation and Projection (UMAP) is used for dimensionality reduction, providing a 3D visualization of document clusters. UMAP preserves the local and global structure of the data, allowing for a meaningful representation of complex high-dimensional data.

3. **3D Document Clusters Visualization:**
   - A Plotly 3D scatter plot displays the document clusters, colored by dominant topics. Users can interact with the plot to explore how documents are grouped based on thematic content, providing insights into the dataset's structure.
""")


# Create a Streamlit app
# st.header('Document Clusters by Dominant Topic (UMAP 3D)')

new_column_names = {
    'title': 'Title',
    'authors': 'Authors',
    'dominant_topic_text_short': 'Topic',
    'bias': 'Bias',
}

# Rename columns
main_list.rename(columns=new_column_names, inplace=True)

mapping = {0: 'Center', 1: 'Left', 2: 'Right'}

main_list['Bias'] = main_list['Bias'].map(mapping)

main_list = main_list.dropna(subset=["text"])
main_list = main_list.drop_duplicates(subset=["Title", "Authors", "top_image"])
main_list = main_list.drop_duplicates().reset_index()

def nmf_topic_modeling(documents, n_topics=30, n_top_words=20):
    # Vectorize the documents
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(documents)

    # Fit NMF model
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf)
    
    return nmf, tfidf

# Run NMF topic modeling
nmf_model, tfidf_matrix = nmf_topic_modeling(main_list['text'], n_topics=30)

# Get document-topic matrix
doc_topic_matrix = nmf_model.transform(tfidf_matrix)


umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
doc_3d = umap_model.fit_transform(doc_topic_matrix)

# Add the UMAP 3D representation to the DataFrame
main_list['x'] = doc_3d[:, 0]
main_list['y'] = doc_3d[:, 1]
main_list['z'] = doc_3d[:, 2]


# Create the Plotly 3D scatter plot
fig = px.scatter_3d(main_list, 
                    x='x', 
                    y='y', 
                    z='z', 
                    color='Topic', 
                    hover_data=['Title', 'Authors', 'Topic', 'Bias'],
                    # title='Document Clusters by Dominant Topic (UMAP 3D)',
                    labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2', 'z': 'UMAP Component 3'})

# Customize plot layout
fig.update_layout(
    #width=1200,  # Width of the plot in pixels
    height=1000,  # Height of the plot in pixels
    legend_title_text='Topic:',  # Change the legend title
    legend=dict(
        font=dict(
            size=16,  # Change the legend text size
        )
    )

)

# Customize marker size
fig.update_traces(marker=dict(size=3))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
# Import Packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
from sklearn.decomposition import NMF
from wordcloud import WordCloud


sns.set_theme()
import joblib
import spacy
import streamlit as st
from newspaper import Article
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words("english"))
import time

import torch
from newspaper import Config
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

# Set up the user agent for the newspaper library
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0"
)


# Set up the newspaper library configuration
config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10

# Set up the Streamlit page configuration
st.set_page_config(page_title="Article Recommendations", page_icon=":bar_chart:", layout="wide")
st.title("News Article Analysis:")


st.markdown("""
This News Article Analysis Dashboard is an interactive tool designed to provide a comprehensive analysis of news articles by assessing their sentiment, political bias, topic identification, and credibility. This dashboard leverages advanced machine learning models and natural language processing techniques to deliver insights into the content and context of news articles.

### Key Features:

1. **Input a News Article URL:**
   - Simply enter a URL of a news article to begin the analysis. The dashboard will automatically download and parse the content for further evaluation.

2. **Article Sentiment Analysis:**
   - Utilizing the VADER sentiment analyzer, the dashboard evaluates the overall sentiment of the article, categorizing it as Positive, Neutral, or Negative. This analysis helps users understand the emotional tone of the article.

3. **Political Bias Detection:**
   - A trained machine learning model is employed to predict the political bias of the article. The bias is classified as Left, Center, or Right, providing insights into the potential slant of the content.

4. **Fake News Detection:**
   - The dashboard incorporates a model to assess the authenticity of the article, determining whether it is likely to be Fake or Real. This feature helps users in identifying unreliable or misleading information.

5. **Topic Identification:**
   - By applying Non-negative Matrix Factorization (NMF) on TF-IDF vectors, this dashboard identifies the dominant topic within the article. Topics range from "U.S. Politics" to "Artificial Intelligence and Technology," giving users a clear understanding of the article's main focus.

6. **Article Summary:**
   - A BART summarization model is used to generate concise summaries of the article, providing a quick overview of the content without reading the full text.

7. **Alternative Perspectives:**
   - Users can explore different perspectives on the same topic by viewing recommendations for similar articles. These recommendations are categorized based on political bias (Left, Center, Right) and ranked by similarity score, allowing users to access diverse viewpoints.
""")



st.divider()

# Load the main articles list
# main_list = pd.read_csv("main_articles_list_v2.csv")
main_list = pd.read_csv("articles_with_all_test.csv")

# Load the trained models
tfidf = joblib.load("trained_models/tfidf_vectorizer.joblib")
model = joblib.load("trained_models/xgboost_model.joblib")


# Set up the Streamlit sidebar
news_article = st.text_input("Input a News Article URL: ", "https://www.example.com")


# Download and parse the news article
first_article = Article(url=news_article, config=config)
first_article.download()
first_article.parse()


def predict_party(text):
    """Predict the political bias of the text using the trained model."""
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)
    # Uncomment below for probability distribution for XGBoost
    # y_pred_proba = model.predict_proba(text_tfidf)
    # print(y_pred_proba)
    return "Left" if prediction[0] == 1 else "Right" if prediction[0] == 2 else "Center"


# Set up the VADER sentiment analyzer
vader_sentiment = SentimentIntensityAnalyzer()


def classify_sentiment(text):
    """Classify the sentiment of the text using the VADER sentiment analyzer."""
    # Get the sentiment scores
    sentiment_scores = vader_sentiment.polarity_scores(text)
    compound = sentiment_scores["compound"]

    # Classify sentiment
    if compound > 0.5:
        return "Positive"
    elif compound >= -0.5:
        return "Neutral"
    else:
        return "Negative"


# Set up the Streamlit columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    if first_article.authors != []:
        st.metric(label="Author:", value=first_article.authors[0])
    else:
        st.metric(label="Author:", value="No Author Found")


with col2:
    sentiment = classify_sentiment(first_article.text)
    st.metric(label="Article Sentiment:", value=sentiment)


with col3:
    new_article = first_article.text
    predicted_party = predict_party(new_article)
    st.metric(label="Political Bias Detected:", value=predicted_party)


with col4:
    tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")

    model = AutoModelForSequenceClassification.from_pretrained(
        "hamzab/roberta-fake-news-classification"
    )

    def predict_fake(title, text):
        input_str = (
            "<title>" + first_article.title + "<content>" + first_article.text + "<end>"
        )
        input_ids = tokenizer.encode_plus(
            input_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        with torch.no_grad():
            output = model(
                input_ids["input_ids"].to(device),
                attention_mask=input_ids["attention_mask"].to(device),
            )
        return dict(
            zip(
                ["Fake", "Real"],
                [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])],
            )
        )

    real_fake = predict_fake(first_article.title, first_article.text)

    real_fake = "Fake" if real_fake["Fake"] > real_fake["Real"] else "Real"

    st.metric(label="Fake News:", value=real_fake)

st.divider()


# Display the article title
st.subheader("Title:")
st.write(first_article.title)

st.subheader("Identified Topic:")
nmf_tfidf = joblib.load("trained_models/nmf_tfidf_vectorizer.joblib")
nmf_model = joblib.load("trained_models/nmf_model.joblib")

topic_tfidf = nmf_tfidf.transform([first_article.text])
topic_distribution = nmf_model.transform(topic_tfidf)

topic_mapping = {
    0: "U.S. Politics",
    1: "General Opinions",
    2: "Trump and Republican Politics",
    3: "Israel-Palestine Conflict",
    4: "Loans and Mortgages",
    5: "Kamala Harris and Vice Presidency",
    6: "U.S. Supreme Court",
    7: "Crime and Policing",
    8: "Olympic Games",
    9: "Weather and Heat",
    10: "Russia-Ukraine Conflict",
    11: "Artificial Intelligence and Technology",
    12: "French Politics",
    13: "General and Celebrity News",
    14: "Secret Service and Security",
    15: "Sports and Athletes",
    16: "U.S. Elections",
    17: "EU General News",
    18: "Abortion and Women’s Rights",
    19: "Microsoft and Cypersecurity",
    20: "Criminal Justice",
    21: "Elon Musk and Tesla",
    22: "Hurricanes and Storms",
    23: "U.S. Political Figures",
    24: "Inflation and Economy",
    25: "Conservative Media",
    26: "UK Politics",
    27: "Health and Medical Research",
    28: "Political Debates",
    29: "Immigration and Border Issues",
}

dominant_topic = topic_distribution.argmax(axis=1)
#st.write("Topic Distribution:", topic_distribution)
st.write(topic_mapping.get(dominant_topic[0]))





st.subheader("Article Summary:")


# Set up the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
tokens = tokenizer.tokenize(first_article.text)
truncated_tokens = tokens[:1022]
truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
summary = summarizer(truncated_text, max_length=200, min_length=30, do_sample=False)


def summary_stream(summary_text):
    """Stream the summary text in chunks."""
    chunk_size = 10
    for i in range(0, len(summary_text), chunk_size):
        yield summary_text[i : i + chunk_size]
        time.sleep(0.025)


st.write_stream(summary_stream(summary[0]["summary_text"]))
st.divider()


nlp = spacy.load("en_core_web_sm")

# Display the article text
main_list = main_list.dropna(subset=["text"])
main_list = main_list.drop_duplicates(subset=["title", "authors", "top_image"])
main_list = main_list.drop_duplicates().reset_index()


def extract_entities(text):
    doc = nlp(text)
    return " ".join([ent.text for ent in doc.ents])


def hash_spacy_pipeline(obj):
    return hash(obj.to_bytes())

from concurrent.futures import ThreadPoolExecutor

def calculate_similarity_with_single_document(documents, single_document):
    """
    Calculates the cosine similarity between a list of documents and a single document.

    Parameters:
    - documents (list): A list of documents to compare with the single document.
    - single_document (str): The single document to compare with the list of documents.

    Returns:
    - cosine_similarities (numpy.ndarray): An array of cosine similarity scores between the single document and each document in the list.
    """

    # entity_docs = []
    # total_docs = len(documents)

    # # Set up Streamlit progress bar and status text
    # progress_text = "Operation in progress. Please wait..."
    # my_bar = st.progress(0, text=progress_text)

    # for i, doc in enumerate(documents):
    #     entity_docs.append(extract_entities(doc))
    #     my_bar.progress((i + 1) / total_docs, text=progress_text)

    single_entity_doc = extract_entities(single_document)
    # entities_list = list(dict.fromkeys(single_entity_doc.split(' ')))
    # filtered_entities_list = [word for word in entities_list if word.lower() not in stop_words]
    # filtered_entities_list
    combined_docs = list(documents) + [single_entity_doc]

    vectorizer = TfidfVectorizer().fit_transform(combined_docs)
    single_doc_vector = vectorizer[-1]
    cosine_similarities = cosine_similarity(single_doc_vector, vectorizer[:-1])

    # my_bar.empty()

    return cosine_similarities.flatten()


single_document = first_article.text

filtered_main_list = main_list.fillna({"entities": ""})

similarities = calculate_similarity_with_single_document(
    filtered_main_list["entities"], single_document
)


results = []

for idx, score in enumerate(similarities):
    result = {
        "Article": idx,
        "Document1_Bias": main_list.loc[idx, "bias"],
        "Similarity_Score": score,
    }
    results.append(result)

results_df = pd.DataFrame(results)


st.header("Find another view on this topic:")


num_articles = st.slider(
    "How many article recommendations would you like to see?", 0, 10, 5
)


clean_results_df = results_df.groupby("Document1_Bias", group_keys=False).apply(
    lambda x: x.sort_values("Similarity_Score", ascending=False).head(num_articles),
    include_groups=True,
)


# col1, col2, col3 = st.columns(3)

# with col1:
st.header("Left:")
left_indices = clean_results_df[clean_results_df["Document1_Bias"] == 1]["Article"]
left_links = main_list.loc[
    left_indices, ["top_image", "title", "authors", "sentiment", "publish_date", "article"]
].merge(
    clean_results_df[["Similarity_Score"]],
    left_index=True,
    right_index=True,
    how="left",
)
st.data_editor(
    left_links,
    column_config={
        "top_image": st.column_config.ImageColumn(
            "Preview", help="Preview screenshots"
        ),
        "article": st.column_config.LinkColumn("Article Link"),
        "Similarity_Score": st.column_config.ProgressColumn(
            "Similarity Score",
            help="How close the articles are in terms of topic.",
            # format='{:.2%}'.format,
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# with col2:
st.header("Center:")
center_indices = clean_results_df[clean_results_df["Document1_Bias"] == 0][
    "Article"
]
center_links = main_list.loc[
    center_indices, ["top_image", "title", "authors", "sentiment", "publish_date", "article"]
].merge(
    clean_results_df[["Similarity_Score"]],
    left_index=True,
    right_index=True,
    how="left",
)
st.data_editor(
    center_links,
    column_config={
        "top_image": st.column_config.ImageColumn(
            "Preview", help="Preview screenshots"
        ),
        "article": st.column_config.LinkColumn("Article Link"),
        "Similarity_Score": st.column_config.ProgressColumn(
            "Similarity Score",
            help="How close the articles are in terms of topic.",
            # format='{:.2%}'.format,
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# with col3:
st.header("Right:")
right_indices = clean_results_df[clean_results_df["Document1_Bias"] == 2]["Article"]
right_links = main_list.loc[
    right_indices, ["top_image", "title", "authors", "sentiment", "publish_date", "article"]
].merge(
    clean_results_df[["Similarity_Score"]],
    left_index=True,
    right_index=True,
    how="left",
)
st.data_editor(
    right_links,
    column_config={
        "top_image": st.column_config.ImageColumn(
            "Preview", help="Preview screenshots"
        ),
        "article": st.column_config.LinkColumn("Article Link"),
        "Similarity_Score": st.column_config.ProgressColumn(
            "Similarity Score",
            help="How close the articles are in terms of topic.",
            # format='{:.2%}'.format,
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
    use_container_width=True,
)












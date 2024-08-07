import newspaper
import nltk
import pandas as pd
import seaborn as sns
from langdetect import detect
from newspaper import Config, news_pool
from newspaper.article import ArticleException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

nltk.download("wordnet")
nltk.download("omw-1.4")
import ast

import joblib
import newspaper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sns.set()

nltk.download("stopwords")
en_stopwords = stopwords.words("english")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0"
)

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10


sites_df = {
    "link": [
        "https://www.foxnews.com/politics",
        "https://www.breitbart.com/politics/",
        "https://www.theamericanconservative.com/category/politics/",
        "https://www.oann.com/",
        "https://fortune.com/the-latest/",
        "https://www.forbes.com/?sh=4c70701d2254",
        "https://www.cbsnews.com/",
        "https://www.bbc.com/news",
        "https://www.cnn.com/politics",
        "https://www.politico.com/",
        "https://www.vox.com/",
        "https://www.msnbc.com/",
        "https://www.businessinsider.com/politics",
        "https://www.cnn.com/",
        "https://www.nytimes.com/",
        "https://www.nbcnews.com/",
        "https://www.usatoday.com/",
        "https://www.washingtonpost.com/",
        "https://www.wsj.com/",
        "https://www.foxbusiness.com/",
        "https://www.bloomberg.com/",
        "https://www.axios.com/",
        "https://www.reuters.com/",
        "https://www.apnews.com/",
        "https://www.npr.org/",
        "https://www.pbs.org/",
        "https://www.buzzfeednews.com/",
        "https://www.huffpost.com/",
        "https://www.theguardian.com/us",
        "https://www.aljazeera.com/",
        "https://www.independent.co.uk/us",
        "https://www.thedailybeast.com/",
        "https://www.motherjones.com/",
        "https://www.thenation.com/",
        "https://www.yahoo.com/news/",
        "https://www.dailywire.com/",
        "https://www.nationalreview.com/",
        "https://www.thefederalist.com/",
        "https://www.theblaze.com/",
    ],
    "name": [
        "Fox News",
        "Breitbart",
        "The American Conservative",
        "OANN",
        "Fortune",
        "Forbes",
        "CBS News",
        "BBC News",
        "CNN Politics",
        "Politico",
        "Vox",
        "MSNBC",
        "Business Insider",
        "CNN",
        "The New York Times",
        "NBC News",
        "USA Today",
        "The Washington Post",
        "The Wall Street Journal",
        "Fox Business",
        "Bloomberg",
        "Axios",
        "Reuters",
        "Associated Press",
        "NPR",
        "PBS",
        "Buzzfeed News",
        "HuffPost",
        "The Guardian",
        "Al Jazeera",
        "The Independent",
        "The Daily Beast",
        "Mother Jones",
        "The Nation",
        "Yahoo News",
        "Daily Wire",
        "National Review",
        "The Federalist",
        "The Blaze",
    ],
    "clean_name": [
        "fox_news",
        "breitbart",
        "the_american_conservative",
        "oann",
        "fortune",
        "forbes",
        "cbs_news",
        "bbc_news",
        "cnn_politics",
        "politico",
        "vox",
        "msnbc",
        "business_insider",
        "cnn",
        "ny_times",
        "nbc_news",
        "usa_today",
        "washington_post",
        "wsj",
        "fox_business",
        "bloomberg",
        "axios",
        "reuters",
        "ap_news",
        "npr",
        "pbs",
        "buzzfeed_news",
        "huffpost",
        "theguardian",
        "aljazeera",
        "independent",
        "daily_beast",
        "mother_jones",
        "the_nation",
        "yahoo_news",
        "daily_wire",
        "national_review",
        "the_federalist",
        "the_blaze",
    ],
    "affiliation": [
        "Right",
        "Right",
        "Right",
        "Right",
        "Center",
        "Center",
        "Center",
        "Center",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Right",
        "Right",
        "Center",
        "Center",
        "Center",
        "Center",
        "Center",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Left",
        "Right",
        "Right",
        "Right",
        "Right",
    ],
}

sites = pd.DataFrame(sites_df)


news_sources = []

for i, site in enumerate(tqdm(sites["link"], desc="Building sites")):
    try:
        paper = newspaper.build(site, config=config, memoize_articles=True)
        news_sources.append(paper)
    except Exception as e:
        print(f"Failed to build newspaper for site {site}: {e}")
        continue

print("Downloading Articles...")
news_pool.set(news_sources, threads_per_source=2)  # (3*2) = 6 threads total
news_pool.join()
print("Download Complete")

articles_data = []

for i, news_site in enumerate(
    tqdm(news_sources, desc=f"Processing articles", leave=False)
):
    article = news_site.articles
    for link in article:
        try:
            link.parse()
            articles_data.append(
                {
                    "title": link.title,
                    "authors": link.authors,
                    "publish_date": link.publish_date,
                    "text": link.text,
                    "top_image": link.top_image,
                    "article": link.url,
                    "site": sites.loc[i, "clean_name"],
                    "site_name": sites.loc[i, "name"],
                    "bias": sites.loc[i, "affiliation"],
                }
            )
        except ArticleException as e:
            print(f"ArticleException for {link.url}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected exception for {link.url}: {e}")
            continue


all_news_articles = pd.DataFrame(articles_data)


news_filtered = all_news_articles.dropna(subset=["authors"])

news_filtered = news_filtered[news_filtered["authors"].apply(lambda x: len(x) > 0)]

df_filtered = news_filtered.reset_index(drop=True)

df_filtered["clean_text"] = df_filtered["text"].apply(lambda x: str(x).lower())


for i, row in enumerate(df_filtered["clean_text"]):
    df_filtered.loc[i, "clean_text"] = " ".join(
        [word for word in row.split() if word not in en_stopwords]
    )


lemmatizer = WordNetLemmatizer()

for i, row in enumerate(df_filtered["clean_text"]):
    df_filtered.loc[i, "clean_text"] = " ".join(
        [lemmatizer.lemmatize(word) for word in row.split()]
    )

df_filtered["clean_text"] = df_filtered["clean_text"].str.strip()

df_filtered["clean_text"] = (
    df_filtered["clean_text"].str.strip().str.replace(r"[^\w\s]", "", regex=True)
)


tfidf = joblib.load("trained_models/tfidf_vectorizer.joblib")
model = joblib.load("trained_models/xgboost_model.joblib")


def predict_party(text):
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)
    # Uncomment below for probability distribution for XGBoost
    # y_pred_proba = model.predict_proba(text_tfidf)
    # print(y_pred_proba)
    return "Left" if prediction[0] == 1 else "Right" if prediction[0] == 2 else "Center"


vader_sentiment = SentimentIntensityAnalyzer()


def classify_sentiment(text):
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


for i, row in tqdm(
    df_filtered.iterrows(),
    total=df_filtered.shape[0],
    desc="Predicting party and sentiment",
):
    text = df_filtered.loc[i, "text"]
    df_filtered.loc[i, "party"] = predict_party(text)
    df_filtered.loc[i, "sentiment"] = classify_sentiment(text)


encoded = df_filtered.copy()

label_encoder = LabelEncoder()
encoded["bias"] = label_encoder.fit_transform(encoded["bias"])

tqdm.pandas()

encoded["language"] = encoded["text"].progress_apply(
    lambda x: detect(x) if x.strip() else "unknown"
)

encoded = encoded[encoded["language"] == "en"].reset_index(drop=True)

print(f"Shape of encoded before concatenation: {encoded.shape}")

main_list = pd.read_csv("main_articles_list_v2.csv")

print(f"Shape of main_list after loading: {main_list.shape}")

stacked_df = pd.concat([main_list, encoded], axis=0).reset_index(drop=True)

print(f"Shape of stacked_df after concatenation: {stacked_df.shape}")

stacked_df.to_csv("main_articles_list_v2.csv", index=False)


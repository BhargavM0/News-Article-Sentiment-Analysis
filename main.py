import streamlit as st
import requests
from datetime import datetime
from collections import Counter

import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except:
    nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("en_core_web_sm")
from PIL import Image

# Retrieves 10 news articles based on search query. It uses the NewsAPI to search for articles (I used a free api key)
NEWSAPI_URL = "https://newsapi.org/v2/everything"
API_KEY = "9c67e212a11d44129d0efc4f28724708"

#Initializes sentiment analyzer from nltk
sia = SentimentIntensityAnalyzer()

#Top 50 verified news sources
verified_sources = [
  #Major US News Outlets
  "bbc-news", "cnn", "the-new-york-times", "the-washington-post",
  "the-wall-street-journal", "usa-today", "abc-news", "cbs-news",
  "nbc-news", "fox-news", "msnbc", "npr", "politico",
  "associated-press", "reuters", "bloomberg",

  #Tech/Business News Outlets
  "techcrunch", "the-verge", "wired", "engadget", "ars-technica",
  "business-insider", "fortune", "forbes", "cnbc",

  #International News Outlets
  "al-jazeera-english", "sky-news", "independent", "the-guardian-uk",
  "the-telegraph", "financial-times", "the-economist", "times-of-india",
  "hindustan-times", "globe-and-mail", "le-monde", "der-spiegel",
  "reuters", "vice-news",

  #Science/Other News Outlets
  "scientific-american", "national-geographic", "new-scientist",
  "medical-news-today", "nature", "the-hill",

  #Regional News Outlets
  "chicago-tribune", "la-times", "miami-herald", "boston-globe",
  "seattle-times", "houston-chronicle", "dallas-morning-news"
]

#Function to retrieve articles based on query. Sorts by most recent
def getArticles(api_key, query, result_count = 100, verifiedSources = False):
  parameters = {
      "q": query,
      "pageSize": result_count,
      "sortBy": "publishedAt",
      "language": "en",
      "apiKey": api_key
  }

  #In case user selects verified sources only
  if verifiedSources:
    parameters["sources"] = ",".join(verified_sources)

  response = requests.get(NEWSAPI_URL, params = parameters)
  return response.json().get("articles", [])

#Helpher function to reformat date/time
def formatDT(isoDate):
  try:
    dt = datetime.strptime(isoDate, "%Y-%m-%dT%H:%M:%SZ")

    return dt.strftime("%B %d at %I:%M %p")
  except:
    return isoDate

#Utilizes the nltk library to analyze an article and give a rating of Positive/Negative/Neutral
def analyzeArticle(article):
  if not article:
    return ""

  fullText = " ".join([article.get("title", ""), article.get("description", ""), article.get("content", "")])

  try:
    sentiment = sia.polarity_scores(fullText)
    compound = sentiment["compound"]
  except:
    return "Neutral"

  if compound >= 0.1:
    connotation =  "Positive"
  elif compound <= -0.1:
    connotation = "Negative"
  else:
    connotation = "Neutral"


  return connotation

#Helper function to list sentiment of
def sentimentMarkdown(article):
    connotation = analyzeArticle(article)

    color_map = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "gray"
    }

    color = color_map.get(connotation, "gray")

    st.markdown(
        f"<span style='color:{color}; font-weight:bold'>Sentiment Classification: {connotation}</span>",
        unsafe_allow_html=True
    )

#This function generates a wordcloud based on the content of the article
def generateWordCloud(article):
  content = article.get("content", " ")

  if content:
    wordcloud = WordCloud(width=400, height=400, background_color="white").generate(content)

  img = wordcloud.to_image()
  st.image(img, width=250)

#This function creates an entity cloud to display important entities in the article
def generateEntityCloud(article):
    content = article.get("content", " ")
    
    if not content:
        return

    doc = nlp(content)
    
    importantentities = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]
    
    entities = [entity.text for entity in doc.ents if entity.label_ in importantentities]
    
    if not entities:
        st.write("No named entities in article")
        return
    
    entitiesCount = Counter(entities)
    commonEntities = entitiesCount.most_common(5)
    
    names, counts = zip(*commonEntities)
    
    fig, ax = plt.subplots(figsize=(4,2))
    ax.barh(names, counts, color="blue")
    ax.set_xlabel("Frequency")
    ax.set_title("Top Entities")
    ax.set_xticks(range(0, max(counts)+1,1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    
    

#Main function. This is what runs when streamlit starts
def main():

  st.title("News Aggregator AI Agent")

  st.markdown("Please enter news preferences (seperated by commas): ")
  preferencesInput = st.text_input("Input as many preferences as youd like...")

  #Options for modifying the search
  with st.container():
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
      customSources = st.checkbox("Only list verified sources", help ="Retrives articles from top 50 English news sources:\n"+", ".join(verified_sources))

    with col2:
      with st.popover("Search Settings"):
        resultCount = st.number_input(
            "Select how many articles you'd like to see: ",
            min_value = 1,
            max_value = 100,
            value = 10,
            step = 1
            )

  if preferencesInput:
    preferences = [term.strip() for term in preferencesInput.split(",") if term.strip()]

    for preference in preferences:

      st.subheader(f"News Articles with {preference}")
      articles = getArticles(API_KEY, preference, verifiedSources=customSources)[:resultCount]

      if articles:

        #Iterates through articles, retrieving and formatting data
        for article in articles:
          title = article.get("title", "CANNOT RETRIEVE TITLE")
          url = article.get("url", "N/A")
          source = article.get("source", {}).get("name", "CANNOT RETRIEVE SOURCE")
          publishedDate = article.get("publishedAt", "")
          author = article.get("author")

          date_clean = formatDT(publishedDate)

          #Popup on button click, for easier visibility
          with st.popover(f"{title} ({source})"):
            st.subheader(title)
            st.markdown(f"*Published by {author if author else 'Unknown'} via {source}*", unsafe_allow_html=True)
            st.write(f"Published on {date_clean}")
            st.markdown(f"[Read Article]({url})", unsafe_allow_html=True)
            sentimentMarkdown(article)
            generateWordCloud(article)
            generateEntityCloud(article)


      else:
        st.write("NO ARTICLES FOUND")

      st.markdown("---")


main()

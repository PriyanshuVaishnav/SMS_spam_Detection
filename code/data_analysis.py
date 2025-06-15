import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
import nltk  
import os

# === Safe download only if not exists ===
def download_nltk_data():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk_data()
stopwords = set(STOPWORDS)

class DataAnalysis:
    def __init__(self, df) -> None:
        self.df = df

    def explore_data(self, vis=False):
        print("Head of the data:- \n", self.df.head())
        print("Shape of the data:-", self.df.shape)
        if vis:
            self.explore_data_visualization()

    def explore_data_visualization(self, show_word_cloud_with_specific_labels=False):
        len_ham = len(self.df[self.df["label"] == "ham"])
        len_spam = len(self.df[self.df["label"] == "spam"])
        arr_labels = np.array([len_ham, len_spam])
        labels = ["Ham", "Spam"]

        print("No of ham messages: ", len_ham)
        print("No of spam messages: ", len_spam)
        plt.pie(
            arr_labels, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
        )
        plt.title("Distribution of Ham vs Spam Messages")
        plt.axis("equal")
        plt.show()

        print("Showing the wordcloud :- ")
        self.show_wordcloud()

        if show_word_cloud_with_specific_labels:
            print("Showing the wordcloud for specific labels :- ")
            self.show_wordcloud_specific_to_targets()

    def show_wordcloud(self):
        wordcloud = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            max_font_size=40,
        ).generate(" ".join(self.df["sms_message"].astype(str)))

        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def show_wordcloud_specific_to_targets(self):
        ham_msgs = self.df[self.df["label"] == "ham"]["sms_message"]
        spam_msgs = self.df[self.df["label"] == "spam"]["sms_message"]

        print("Showing the wordcloud for Ham messages:- ")
        wordcloud_ham = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=1,
        ).generate(" ".join(ham_msgs.astype(str)))

        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud_ham, interpolation="bilinear")


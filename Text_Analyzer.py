# Get ready for the Dependencies
import os
import string
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# get the length of files in the directory path
def get_len(path):
    count = 0
    for file in os.listdir(path):
        if file.endswith(".txt"):
            count += 1
    return count

# print(get_len("./Dealy_news"), "files")

def deal_with_files():
    text, index = "", 0
    path = "./Trump_Rally_Speeches"
    count = get_len(path)
    for files in os.listdir(path):
        if count > 0:
            if files.endswith(".txt"):
                file_path = os.path.join(path, files)
            index += 1
            with open(file_path, encoding="utf-8") as f_:
                for lines in f_:
                    lines = lines.strip()
                    text += lines
                    if index == count:  # if the index is equal to the number of files
                        break
        else:
            return 'No files found'
    return text

# put the text into a file
with open("Trump_Rally_Speeches.txt", "w", encoding="utf-8") as f_:
    f_.write(deal_with_files())


def text_cleaning():
    # 1- Load the data
    text = deal_with_files()
    # 1- Convert text to lowercase
    lower_letters = text.lower()
    # 2- Remove punctuation characters
    punctuation_characters = string.punctuation  # Get the punctuation characters
    cleaned_text = lower_letters.translate(str.maketrans('', '', punctuation_characters))
    # 3- Split the string into a list of words.
    tokenized_text = word_tokenize(cleaned_text, 'english')
    # 4- Remove stop words from the list of words
    stop_words = stopwords.words('english')
    final_words = []
    for word in tokenized_text:
        try:
            if word not in stop_words:
                final_words.append(word)
        except TypeError as TE:  # 'LazyCorpusLoader' object is not callable
            print(f'{TE}' + '\n')
    # do a line space if length of line is 7
    counter = 0
    new_text = ""
    for word in final_words:  # for each word in the list
        new_text += word + " "  # add the word to the new text
        counter += 1
        if counter == 18:
            new_text += "\n"  # add a line space
            counter = 0
    return new_text


with open('Trump_Rally_Speeches.txt', 'w', encoding='utf-8') as file:
    file.write(text_cleaning())


# emotions analysis
def emotions_detector():
    # 1- Check up if word in final word list is also present in emotion text
    emotions_list = []
    text = text_cleaning()  # get the cleaned text
    splitting_text = text.split()  # split the text into a list of words
    with open('Emotions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            cleared_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            word, emotion = cleared_line.split(':')
            # print('Word: {}, Emotion: {}'.format(word, emotion))
            for word_emotion in splitting_text:
                if word_emotion == word:
                    emotions_list.append(emotion)

    # 2- Count the number of times each emotion appears in the list
    emotions_count = Counter(emotions_list)
    return emotions_count

# emo = emotions_detector()
# print(emo)

# function to get the sentiment analysis rely on emotion count
def sentiment_analysis():
    emotions_count = emotions_detector()  # get the emotions count
    emotions_count_list = emotions_count.most_common(12)  # get the top 12 emotions
    emotion_list = []
    for emo in emotions_count_list:
        emotion_list.append(emo)
    # print(emotion_list)
    # plot the emotions count and emotions
    emotions_count_list = [x[1] for x in emotion_list]  # get the emotions count
    emotions = [x[0] for x in emotion_list]  # get the emotions names
    plt.bar(emotions, emotions_count_list)  # plot the emotions count
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotions Analysis')
    plt.savefig('Emotions_Analysis.png')
    plt.show()
    # get the sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_analysis_ = []
    for emotion in emotions:
        sentiment_analysis_.append(sid.polarity_scores(emotion))  # get the sentiment analysis for 12 emotions separately
    return sentiment_analysis_  # return the sentiment analysis as a list of dictionaries for each emotion

# print(sentiment_analysis())



# function to tell me if the sentiment analysis is positive or negative
def sentiment_analysis_positive_negative():
    sentiment_analysis_ = sentiment_analysis()  # get the sentiment analysis
    positive_negative_list = []
    for sentiment in sentiment_analysis_:
        if sentiment['compound'] >= 0.05:  # if the sentiment is positive
            positive_negative_list.append('Positive')
        elif sentiment['compound'] < -0.05:  # if the sentiment is negative
            positive_negative_list.append('Negative')
        else:  # if the sentiment is neutral
            positive_negative_list.append('Neutral')
    if positive_negative_list.count('Positive') > positive_negative_list.count('Negative'):
        return 'The Speech was positive effect on the Trump rally'
    elif positive_negative_list.count('Positive') < positive_negative_list.count('Negative'):
        return 'The Speech was negative effect on the Trump rally'
    else:
        return 'The Speech was neutral effect on the Trump rally'
    
            
print(sentiment_analysis_positive_negative())           

# just to read the orginal data 
def original_data():
    docs, index = [], 0
    path = "./Trump_Rally_Speeches"
    count = get_len(path)
    for files in os.listdir(path):
        if count > 0:
            if files.endswith(".txt"):
                file_path = os.path.join(path, files)
            index += 1
            with open(file_path, encoding="utf-8") as f_:
                for lines in f_:
                    lines = lines.strip()
                    docs.append(lines)
                    if index == count:  # if the index is equal to the number of files
                        break
        else:
            return 'No files found'
    return docs

# fucntion sub-precessing the data
def preprocess(text):
     # remove punctuation
    punctuation_characters = string.punctuation  # Get the punctuation characters
    cleaned_text = text.translate(str.maketrans('', '', punctuation_characters))
    # remove digits
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    return cleaned_text

# calc the tf-idf of the words
def tf_idf():
    # get the data
    data_1 = original_data()  # list of words in the speeches
    data_2 = [preprocess(items) for items in data_1]  # list of words in the speeches after preprocessing
    # calc the tf-idf
    vectorizer = TfidfVectorizer()  # create a tf-idf vectorizer
    X = vectorizer.fit_transform(data_2)  # get the tf-idf of the words
    # get the feature names, which are the words
    feature_names = vectorizer.get_feature_names_out()
    # len of the feature names
    len_feature_names = len(feature_names)
    # put the feature names in a list
    feature_names_list = np.array(feature_names)
    # sort the feature names
    feature_names_list = feature_names_list[np.argsort(X.toarray()[0])[::-1]]
    
    unique_feature_names = set(feature_names_list)
    return unique_feature_names,\
    " :: Length of the feature names: {} words".format(len_feature_names)




print(tf_idf())



sys.exit()  # exit the program
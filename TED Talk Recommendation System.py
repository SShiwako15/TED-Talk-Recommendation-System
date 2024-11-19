import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
warnings.filterwarnings('ignore')

data = pd.read_csv('tedx_dataset.csv')
print(data.head())
print(data.shape)

print(data.isnull().sum())

split_date = data['posted'].str.split(' ', expand = True)
data['year'] = split_date[2].astype('int')
data['month'] = split_date[1]

print(data.info())

data['year'].value_counts().sort_index().plot(kind ='bar')
plt.show()

data['detials'] = data['title'] + ' ' + data['details']

data = data[['main_speaker', 'details']]
data.dropna(inplace = True)
data.head()

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    
    for words in str(text).split():
        words = words.lower()
    
        if words not in stop_words:
            imp_words.append(words)

    output = " ".join(imp_words)

    return output

data['details'] = data['details'].apply(lambda text: remove_stopwords(text))
data.head()

def remove_punctuations(text):
    punctuation_list = string.punctuation
    signal = str.maketrans('','', punctuation_list)
    return text.translate(signal)

data['details'] = data['details'].apply(lambda x: remove_punctuations(x))
data.head()

det_corp = " ".join(data['details'])

plt.figure(figsize=(30,30))
wc = WordCloud(max_words = 1000, width = 800, height = 400).generate(det_corp)
plt.axis('off')
plt.imshow(wc)
plt.show()

vectorizer = TfidfVectorizer(analyzer = 'word')
vectorizer.fit(data['details'])

def get_similarities(talk_content, df = data):
    talk_array1 = vectorizer.transform(talk_content).toarray()

    sim = []
    pea = []
    for idx,row in df.iterrows():
        details = row['details']

        talk_array2 = vectorizer.transform(df[df['details'] == details]['details']).toarray()
        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

        pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]
        sim.append(cos_sim)
        pea.append(pea_sim)

    return sim, pea

def recommend_talks(talk_content, data = data):
    data['cos_sim'], data['pea_sim'] = get_similarities(talk_content)

    data.sort_values(by = ['cos_sim', 'pea_sim'], ascending = [False, False], inplace = True)

    display(data[['main_speaker', 'details']].head())

talk_content = ['Time Management and working\
hard to become successful in life']
recommend_talks(talk_content)
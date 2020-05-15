import rbm2
import numpy as np

import re
import math

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize, LineTokenizer, PorterStemmer, FreqDist
from nltk.stem import WordNetLemmatizer

stopwords = stopwords.words('english')
stopwords.extend(["though", "yet", "'s", "n't", "'ve", "'", "-", "--"])
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

them_words = ['stori', 'novel', 'reader', 'fiction', 'tale', 'charact', 'love', 'fan', 'book', 'read', 'entertain', 'etc', 'one', 'mysteri', 
    'time', 'like', 'plot', 'work', 'thriller', 'write', 'histor', 'may', 'author', 'end', 'find', 'life', 'debut', 'seri', 'famili', 'even', 
    'complex', 'portrait', 'woman', 'compel', 'appeal', 'romanc', 'short', 'world', 'collect', 'interest', 'often', 'well', 'enjoy', 'emot', 
    'literari', 'new', 'despit', 'narr', 'beauti', 'prose', 'writer', 'take', 'keep', 'best', 'way', 'sometim', 'better', 'mani', 'power', 'promis', 
    'genr', 'studi', 'play', 'enough', 'moral', 'mind', 'thought', 'drama', 'part', 'intrigu', 'provid', 'overli', 'point', 'look', 'action', 
    'anoth', 'offer', 'much', 'dark', 'funni', 'storytel', 'craft', 'surpris', 'insight', 'reflect', 'magic', 'human', 'engag', 'poignant', 'set', 
    'fun', 'old', 'provoc', 'hero', 'plenti']

def create_features(file_name):
    rfile = open(file_name, encoding='utf-8')
    review_text = rfile.read()
    rfile.close()
    
    sent_list = []
    firstlast_list = []
    len_list = []
    them_freq_list = []
    book_freq_list = []

    reviews = review_text.split("¶")
    
    for r in range(len(reviews)):
        sents = [nltk.sent_tokenize(p) for p in LineTokenizer(blanklines='discard').tokenize(reviews[r])]

        for p in range(len(sents)):
            for s in range(len(sents[p])):
                sent_list.append(sents[p][s])

                if s==0 or s==len(sents[p])-1:
                    firstlast_list.append(1)
                else:
                    firstlast_list.append(0)
                            
                words = clean_text(sents[p][s])

                if len(words) > 4:
                    len_list.append(len(words))
                else:
                    len_list.append(0)
                
                tcount = 0
                for x in range(len(them_words)):
                    if them_words[x] in words:
                        tcount += 1
                if len(words) > 0:
                    them_freq_list.append(tcount/len(words)/20)
                else:
                    them_freq_list.append(0)

                book_words = freq(review_text)
                book_words = np.setdiff1d(book_words, them_words)

                bcount = 0
                for x in range(len(book_words)):
                    if book_words[x] in words:
                        bcount += 1
                if len(words) > 0:
                    book_freq_list.append(bcount/len(words))
                else:
                    book_freq_list.append(0)
            
                   
    size = len(sent_list)
    position_list = [1 if x==0 or x==size-1 else math.cos(2*math.pi*x/size)/4+0.5 for x in range(size)]
    tfidf = tf_idf(sent_list)
    tfidf_list = [0 if len_list[x]==0 else tfidf[x]/len_list[x]/20 for x in range(size)]

    #feature_list = [firstlast_list, len_list, them_freq_list, position_list, tfidf_list]

    feature_list = np.column_stack((firstlast_list, len_list, them_freq_list, book_freq_list, position_list, tfidf_list))

    return (sent_list, feature_list)
           
def clean_text(sent):
    sent = sent.lower()
    sent = re.sub(r"[^\w\d'\s\-]+", '', sent)
    words = nltk.word_tokenize(sent)

    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    words = [porter_stemmer.stem(word) for word in words]

    return words
               
def freq(text):
    text = text.lower()
    text = re.sub(r"[^\w\d'\s\-]+", '', text)
    words = nltk.word_tokenize(text)

    words = [word for word in words if word not in stopwords]
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    words = [porter_stemmer.stem(word) for word in words]

    word_freq = FreqDist(words)
    return [x[0] for x in word_freq.most_common(50)]

def tf_idf(sentences):
    vectorizer = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=10000)
    wc_vector = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    transformer.fit(wc_vector)

    freq_matrix = vectorizer.transform(sentences)
    tfidf_matrix = transformer.transform(freq_matrix)
    tfidf_dense = tfidf_matrix.todense()

    sum_matrix = [sum(tfidf_dense.tolist()[x]) for x in range(len(sentences))]
    return sum_matrix


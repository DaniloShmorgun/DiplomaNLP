## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for analysis
import re
import langdetect 
import nltk
import wordcloud
import contractions

## for sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob



## for machine learning
from sklearn import preprocessing,  feature_extraction,metrics 

## for W2V and textRank
import gensim
import gensim.downloader as gensim_api

## for summarization

import difflib

'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()




###############################################################################
#                  TEXT ANALYSIS                                              #
###############################################################################
'''
Plot univariate and bivariate distributions.
'''
def plot_distributions(dtf, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if dtf[x].nunique() <= max_cat:
            if top is None:
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(dtf[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

    ## bivariate
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=15)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="", ax=ax[1])
        ax[0].set(title="histogram")
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].set(title="density")
        ax[1].grid(True)
    plt.show()



'''
Detect language of text.
'''
def add_detect_lang(data, column):
    dtf = data.copy()
    dtf['lang'] = dtf[column].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")
    return dtf



'''
Compute different text length metrics.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with 2 new columns
'''
def add_text_length(data, column):
    dtf = data.copy()
    columns = ['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_lenght']

    if not any(col in dtf.columns.tolist() for col in columns): 
        dtf['word_count'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))) )
        dtf['char_count'] = dtf[column].apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))) )
        dtf['sentence_count'] = dtf[column].apply(lambda x: len(nltk.sent_tokenize(str(x))) )
        dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
        dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']
        print(dtf[['char_count','word_count','sentence_count','avg_word_length','avg_sentence_lenght']].describe().T[["min","mean","max"]])
    else:
        dtf['word_count_y'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))) )
        dtf['char_count_y'] = dtf[column].apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))) )
        dtf['sentence_count_y'] = dtf[column].apply(lambda x: len(nltk.sent_tokenize(str(x))) )
        dtf['avg_word_length_y'] = dtf['char_count_y'] / dtf['word_count_y']
        dtf['avg_sentence_lenght_y'] = dtf['word_count_y'] / dtf['sentence_count_y']
        print(dtf[['char_count_y','word_count_y','sentence_count_y','avg_word_length_y','avg_sentence_lenght_y']].describe().T[["min","mean","max"]])

    return dtf



'''
Computes the sentiment using Textblob or Vader.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param algo: string - "textblob" or "vader"
    :param sentiment_range: tuple - if not (-1,1) score is rescaled with sklearn
:return
    dtf: input dataframe with new sentiment column
'''
def add_sentiment(data, column, algo="vader", sentiment_range=(-1,1)):
    dtf = data.copy()
    ## calculate sentiment
    if algo == "vader":
        vader = SentimentIntensityAnalyzer()
        dtf["sentiment"] = dtf[column].apply(lambda x: vader.polarity_scores(x)["compound"])
    elif algo == "textblob":
        dtf["sentiment"] = dtf[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    ## rescaled
    if sentiment_range != (-1,1):
        dtf["sentiment"] = preprocessing.MinMaxScaler(feature_range=sentiment_range).fit_transform(dtf[["sentiment"]])
    print(dtf[['sentiment']].describe().T)
    return dtf



'''
Creates a list of stopwords.
:parameter
    :param lst_langs: list - ["english", "italian"]
    :param lst_add_words: list - list of new stopwords to add
    :param lst_keep_words: list - list words to keep (exclude from stopwords)
:return
    stop_words: list of stop words
'''      
def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))



'''
Preprocess a string.
:parameter
    :param txt: string - name of column containing text
    :param lst_regex: list - list of regex to remove
    :param punkt: bool - if True removes punctuations and characters
    :param lower: bool - if True convert lowercase
    :param slang: bool - if True fix slang into normal words
    :param lst_stopwords: list - list of stopwords to remove
    :param stemm: bool - whether stemming is to be applied
    :param lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
    ## Regex (in case, before cleaning)
    if lst_regex is not None: 
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    ## Clean 
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang 
    txt = contractions.fix(txt) if slang is True else txt
            
    ## Tokenize (convert from string to list)
    lst_txt = txt.split()
                
    ## Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
                
    ## Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    ## Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
            
    ## Back to string
    txt = " ".join(lst_txt)
    return txt



'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    : input dataframe with two new columns
'''
def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, slang=False, lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, slang, lst_stopwords, stemm, lemm))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)



'''
Compute n-grams frequency with nltk tokenizer.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''
def word_freq(corpus, ngrams=[1,2,3], top=10, tail=10, figsize=(10,7)):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams
    
    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dtf_freq = pd.concat([dtf_freq, dtf_n])
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
    dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
    
    ## plot
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax =plt.subplots(1,2, figsize=figsize)
    
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax[0],
                data=dtf_freq.groupby('ngrams')[["ngrams","freq","word"]].head(top))
    ax[0].set(xlabel=None, ylabel=None, title="Most frequent words")
    ax[0].grid(axis="x")
    
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax[1],
                data=dtf_freq.groupby('ngrams')[["ngrams","freq","word"]].tail(tail))
    ax[1].set(xlabel=None, ylabel=None, title="Least frequent words")
    ax[1].grid(axis="x")
    plt.subplots_adjust(hspace = 2.0)
    fig.tight_layout(pad=5.0)
    
    plt.show()
    
    return dtf_freq



'''
Plots a wordcloud from a list of Docs or from a dictionary
:parameter
    :param corpus: list - dtf["text"]
'''
def plot_wordcloud(corpus, max_words=150, max_font_size=35, figsize=(10,10)):
    wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus)) #if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
    fig = plt.figure(num=1, figsize=figsize)
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()



'''
Adds a column with word frequency.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param lst_words: list - ["donald trump", "china", ...]
    :param freq: str - "count" or "tfidf"
:return
    dtf: input dataframe with new columns
'''
def add_word_freq(data, column, lst_words, freq="count"):
    dtf = data.copy()

    ## query
    print("found records:")
    print([word+": "+str(len(dtf[dtf[column].str.contains(word)])) for word in lst_words])
    
    ## vectorizer
    lst_grams = [len(word.split(" ")) for word in lst_words]
    if freq == "tfidf":
        vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    else:
        vectorizer = feature_extraction.text.CountVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    dtf_X = pd.DataFrame(vectorizer.fit_transform(dtf[column]).todense(), columns=lst_words)
    
    ## join
    for word in lst_words:
        dtf[word] = dtf_X[word]
    return dtf




###############################################################################
#                        WORD2VEC (WORD EMBEDDING)                            #
###############################################################################
'''
Create a list of lists of grams with gensim:
    [ ["hi", "my", "name", "is", "Tom"], 
      ["what", "is", "yours"] ]
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
:return
    lst of lists of n-grams
'''
def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    
    ## detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus



'''
Train common bigrams and trigrams detectors with gensim
:parameter
    :param corpus: list - dtf["text"]
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_common_terms: list - ["of","with","without","and","or","the","a"]
    :param min_count: int - ignore all words with total collected count lower than this value
:return
    list with n-grams models and dataframe with frequency
'''
def create_ngrams_detectors(corpus, grams_join=" ", lst_common_terms=[], min_count=5, top=10, figsize=(10,7)):
    ## fit models
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=1, grams_join=grams_join)
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=grams_join.encode(), common_terms=lst_common_terms, 
                                                     min_count=min_count, threshold=min_count*2)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=grams_join.encode(), common_terms=lst_common_terms, 
                                                      min_count=min_count, threshold=min_count*2)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    ## plot
    dtf_ngrams = pd.DataFrame([{"word":grams_join.join([gram.decode() for gram in k]), "freq":v} for k,v in trigrams_detector.phrasegrams.items()])
    dtf_ngrams["ngrams"] = dtf_ngrams["word"].apply(lambda x: x.count(grams_join)+1)
    dtf_ngrams = dtf_ngrams.sort_values(["ngrams","freq"], ascending=[True,False])
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_ngrams.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return [bigrams_detector, trigrams_detector], dtf_ngrams






###############################################################################
#                        TOPIC MODELING                                       #
###############################################################################
'''
Use Word2Vec to get a list of similar words of a given input words list
:parameter
    :param lst_words: list - input words
    :param top: num - number of words to return
    :param nlp: gensim model
:return
    list with input words + output words
'''
def get_similar_words(lst_words, top, nlp=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    lst_out = lst_words
    for tupla in nlp.most_similar(lst_words, topn=top):
        lst_out.append(tupla[0])
    return list(set(lst_out))


'''
Fits Latent Dirichlet Allocation with gensim.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param n_topics: num - number of topics to find
:return
    model and dtf topics
'''
def fit_lda(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], n_topics=3, figsize=(10,7)):
    ## train the lda
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)
    id2word = gensim.corpora.Dictionary(lst_corpus) #map words with an id
    dic_corpus = [id2word.doc2bow(word) for word in lst_corpus]  #create dictionary Word:Freq
    print("--- training ---")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=n_topics, 
                                                random_state=123, update_every=1, chunksize=100, 
                                                passes=10, alpha='auto', per_word_topics=True)
    
    ## output
    lst_dics = []
    for i in range(0, n_topics):
        lst_tuples = lda_model.get_topic_terms(i)
        for tupla in lst_tuples:
            lst_dics.append({"topic":i, "id":tupla[0], "word":id2word[tupla[0]], "weight":tupla[1]})
    dtf_topics = pd.DataFrame(lst_dics, columns=['topic','id','word','weight'])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y="word", x="weight", hue="topic", data=dtf_topics, dodge=False, ax=ax).set_title('Main Topics')
    ax.set(ylabel="", xlabel="importance")
    plt.show()
    return lda_model, dtf_topics


###############################################################################
#                  STRING MATCHING                                            #
###############################################################################
'''
Matches strings with cosine similarity.
:parameter
    :param a: string - ex. "my house"
    :param lst_b: list of strings - ex. ["my", "hi", "house", "sky"]
    :param threshold: num - similarity threshold to consider the match valid
    :param top: num - number of matches to return
:return
    dtf with 1 column = a, index = lst_b, values = cosine similarity scores
'''
def utils_string_matching(a, lst_b, threshold=None, top=None):
    ## vectorizer ("my house" --> ["my", "hi", "house", "sky"] --> [1, 0, 1, 0])
    vectorizer = feature_extraction.text.CountVectorizer()
    X = vectorizer.fit_transform([a]+lst_b).toarray()

    ## cosine similarity (scores a vs lst_b)
    lst_vectors = [vec for vec in X]
    cosine_sim = metrics.pairwise.cosine_similarity(lst_vectors)
    scores = cosine_sim[0][1:]

    ## match
    match_scores = scores if threshold is None else scores[scores >= threshold]
    match_idxs = range(len(match_scores)) if threshold is None else [i for i in np.where(scores >= threshold)[0]] 
    match_strings = [lst_b[i] for i in match_idxs]

    ## dtf
    dtf_match = pd.DataFrame(match_scores, columns=[a], index=match_strings)
    dtf_match = dtf_match[~dtf_match.index.duplicated(keep='first')].sort_values(a, ascending=False).head(top)
    return dtf_match



'''
Vlookup for similar strings.
:parameter
    :param lst_left - array or lst
    :param lst_right - array or lst
    :param threshold: num - similarity threshold to consider the match valid
    :param top: num or None - number of matches to return
:return
    dtf_matches - dataframe with matches
'''
def vlookup(lst_left, lst_right, threshold=0.7, top=1):
    try:
        dtf_matches = pd.DataFrame(columns=['string','match','similarity'])
        for string in lst_left:
            dtf_match = utils_string_matching(string, lst_right, threshold, top)
            dtf_match = dtf_match.reset_index().rename(columns={'index':'match', string:'similarity'})
            dtf_match["string"] = string
            for i in range(len(dtf_match)):
                print(string, " --", round(dtf_match["similarity"].values[i], 2), "--> ", dtf_match["match"].values[i])
            dtf_matches = dtf_matches.append(dtf_match, ignore_index=True, sort=False)
        return dtf_matches[['string','match','similarity']]

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Find the matching substrings in 2 strings.
:parameter
    :param a: string - raw text
    :param b: string - raw text
:return
    2 lists used in to display matches
'''
def utils_split_sentences(a, b):
    ## find clean matches
    match = difflib.SequenceMatcher(isjunk=None, a=a, b=b, autojunk=True)
    lst_match = [block for block in match.get_matching_blocks() if block.size > 20]
    
    ## difflib didn't find any match
    if len(lst_match) == 0:
        lst_a, lst_b = nltk.sent_tokenize(a), nltk.sent_tokenize(b)
    
    ## work with matches
    else:
        first_m, last_m = lst_match[0], lst_match[-1]

        ### a
        string = a[0 : first_m.a]
        lst_a = [t for t in nltk.sent_tokenize(string)]
        for n in range(len(lst_match)):
            m = lst_match[n]
            string = a[m.a : m.a+m.size]
            lst_a.append(string)
            if n+1 < len(lst_match):
                next_m = lst_match[n+1]
                string = a[m.a+m.size : next_m.a]
                lst_a = lst_a + [t for t in nltk.sent_tokenize(string)]
            else:
                break
        string = a[last_m.a+last_m.size :]
        lst_a = lst_a + [t for t in nltk.sent_tokenize(string)]

        ### b
        string = b[0 : first_m.b]
        lst_b = [t for t in nltk.sent_tokenize(string)]
        for n in range(len(lst_match)):
            m = lst_match[n]
            string = b[m.b : m.b+m.size]
            lst_b.append(string)
            if n+1 < len(lst_match):
                next_m = lst_match[n+1]
                string = b[m.b+m.size : next_m.b]
                lst_b = lst_b + [t for t in nltk.sent_tokenize(string)]
            else:
                break
        string = b[last_m.b+last_m.size :]
        lst_b = lst_b + [t for t in nltk.sent_tokenize(string)]
    
    return lst_a, lst_b



'''
Highlights the matched strings in text.
:parameter
    :param a: string - raw text
    :param b: string - raw text
    :param both: bool - search a in b and, if True, viceversa
    :param sentences: bool - if False matches single words
:return
    text html, it can be visualized on notebook with display(HTML(text))
'''
def display_string_matching(a, b, both=True, sentences=True, titles=[]):
    if sentences is True:
        lst_a, lst_b = utils_split_sentences(a, b)
    else:
        lst_a, lst_b = a.split(), b.split()       
    
    ## highlight a
    first_text = []
    for i in lst_a:
        if re.sub(r'[^\w\s]', '', i.lower()) in [re.sub(r'[^\w\s]', '', z.lower()) for z in lst_b]:
            first_text.append('<span style="background-color:rgba(255,215,0,0.3);">' + i + '</span>')
        else:
            first_text.append(i)
    first_text = ' '.join(first_text)
    
    ## highlight b
    second_text = []
    if both is True:
        for i in lst_b:
            if re.sub(r'[^\w\s]', '', i.lower()) in [re.sub(r'[^\w\s]', '', z.lower()) for z in lst_a]:
                second_text.append('<span style="background-color:rgba(255,215,0,0.3);">' + i + '</span>')
            else:
                second_text.append(i)
    else:
        second_text.append(b) 
    second_text = ' '.join(second_text)
    
    ## concatenate
    if len(titles) > 0:
        first_text = "<strong>"+titles[0]+"</strong><br>"+first_text
    if len(titles) > 1:
        second_text = "<strong>"+titles[1]+"</strong><br>"+second_text
    else:
        second_text = "---"*65+"<br><br>"+second_text
    final_text = first_text +'<br><br>'+ second_text
    return final_text


'''
Summarizes corpus with TextRank.
:parameter
    :param corpus: list - dtf["text"]
    :param ratio: length of the summary (ex. 20% of the text)
:return
    list of summaries
'''
def textrank(corpus, ratio=0.2):
    if type(corpus) is str:
        corpus = [corpus]
    lst_summaries = [gensim.summarization.summarize(txt, ratio=ratio) for txt in corpus]
    return lst_summaries



def evaluate_summary(y_test, predicted):
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(y_test, predicted, avg=True)
    score_1 = round(scores['rouge-1']['f'], 2)
    score_2 = round(scores['rouge-2']['f'], 2)
    score_L = round(scores['rouge-l']['f'], 2)
    print("rouge1:", score_1, "| rouge2:", score_2, "| rougeL:", score_2, 
          "--> avg rouge:", round(np.mean([score_1,score_2,score_L]), 2))
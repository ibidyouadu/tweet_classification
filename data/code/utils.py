# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:06:06 2019

@author: JID002
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import re # regular expression
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import subprocess
def convertSVGtoEMF(figname):
    cmd = 'inkscape -z %s.svg -M %s.emf' % (figname, figname)
    result = []
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    for line in process.stdout:
        result.append(line)
    errcode = process.returncode 
    for line in result:
        print(line)
    if errcode is not None:
        raise Exception('cmd %s failed, see above for details', cmd)
        
        
def top_topics(tweets, meaningless_words, top_k):
    tok = WordPunctTokenizer()
    texts = tweets.text
    meaningless_words = meaningless_words + ['hurricane', 'harvey', 'houstonstrong', 'due',
                                            'hurricaneharvey', 'amp', 'houston', 'traffic',
                                           'texas', 'tornado', 'storm', 'cdt', 'texasstrong']
    # stop_words = list(set(stopwords.words('english')))
    meaningfull_list = []
    for text in texts:
        text_words = [x for x in tok.tokenize(text) if not x in meaningless_words]
        # for tweet with at least one meaningfull word
        if len(text_words) > 0:
            if len(text_words) == 1:
                # remove word with less than 2 characters
                if len(text_words[0]) > 2:
                    meaningfull_list.append(text_words[0])
            else:
                text_words = [x for x in text_words if len(x) > 2]
                meaningfull_list.append((" ".join(text_words)).strip())
    
    if len(meaningfull_list) >= 1:
        cvec = CountVectorizer()
        X = cvec.fit_transform(meaningfull_list)
        tn = pd.Series(cvec.get_feature_names())
        tf = np.sum(X, axis=0)
        tf = np.squeeze(np.asarray(tf))
        idx = np.argsort(-tf)
        if len(tn) > top_k:
            top_words = pd.Series.tolist(tn[idx[:top_k]])
        else:
            top_words = pd.Series.tolist(tn[idx])
        top_words = [top_words[ii] + '_' + str(tf[idx[ii]]) for ii in range(len(top_words))]
    else:
        top_words = np.nan
    
    return top_words


def hex2rgb(c):
    return tuple(int(c[i:i+2], 16)/256.0 for i in (1, 3 ,5))


def group_barplot(values, xy_info, labels, save_flag='no'):
    xlabel = xy_info['xlabel']
    ylabel = xy_info['ylabel']
    xticks = xy_info['xticks']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [hex2rgb(colors[i]) for i in range(len(colors))]
    bar_width = 0.25
    r_base = np.arange(len(values.index))
    plt.figure(figsize=(8, 5))
    for ii in range(len(values.columns)):
        r = r_base + ii * bar_width
        plt.bar(r, values.iloc[:, ii], width=bar_width, color=colors[ii]+(0.5,), label=labels[ii])
    xtick_loc = len(values.columns) / 2 - 0.5
    plt.xticks([r_tick + xtick_loc * bar_width for r_tick in range(len(r_base))], xticks)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    if save_flag is not 'no':
        plt.savefig(save_flag + '.svg')
        convertSVGtoEMF(save_flag)


def date_make(item):
    datetime_str = item[4:10] + item[-5:]
    datetime_obj = datetime.datetime.strptime(datetime_str, '%b %d %Y')
    
    return datetime_obj.date()


def lemmatize(x):
    lemmatizer = WordNetLemmatizer()

    return lemmatizer.lemmatize(lemmatizer.lemmatize(x, pos='v'))


def tweet_clean(text):
    '''function to clean each tweet
    - Remove www nad http patterns
    - Remove numbers and symbols
    - Negation handling
    - Lowercase
    - Word stemming
    - Remove words with less than 2 characters
    '''
    stop_words = set(stopwords.words('english'))
    tok = WordPunctTokenizer()
    
#    remove invalid symbol ("\ufffd")
    try:
        bom_removed = text.decode("utf-8-sig").replace(u"\ufffd", "")
    except:
        bom_removed = text
    
#    remove mentioned username and links(https and www) patterns
    user_pat = r'@[A-Za-z0-9_]+'
    http_pat1 = r'http://[^ ]+'
    http_pat2 = r'https://[^ ]+'
    www_pat = r'www.[^ ]+'
    combined_pat = r'|'.join((user_pat, http_pat1, http_pat2, www_pat)) 
    pat_removed = re.sub(combined_pat, '', bom_removed)
    lower_text = pat_removed.lower()
    
#    handle negation patterns
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", 
                     "weren't":"were not","haven't":"have not","hasn't":"has not",
                     "hadn't":"had not","won't":"will not","wouldn't":"would not", 
                     "don't":"do not", "doesn't":"does not","didn't":"did not",
                     "can't":"can not","couldn't":"could not","shouldn't":"should not",
                     "mightn't":"might not", "mustn't":"must not"}
    neg_pat = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    neg_handled = neg_pat.sub(lambda x: negations_dic[x.group()], lower_text)
    
#    remove non-letter characters
    letters_only = re.sub("[^a-zA-Z0-9]", " ", neg_handled)
    
#    token and lemmatize words
    words = [lemmatize(x) for x in tok.tokenize(letters_only)]
#    filter out tokens with less than 1 characters
    words = [x for x in words if len(x) > 1]
#    filter out tokens of stopping words and meaningless words
    meaningless_words = []
    stop_words = set(stopwords.words('english'))
    words = [x for x in words if not x in stop_words]
    
    return (" ".join(words)).strip()


def keyword_filtering(texts, keywords):
    idxs = []
    idx = 0
    for text in texts:
        if not isinstance(text, float):
            words = text.split()
            intersection = [ii for ii in words if ii in keywords]
            if len(intersection) > 0:
                idxs.append(idx)
        idx += 1
        
    return np.array(idxs, dtype=int)
            
    
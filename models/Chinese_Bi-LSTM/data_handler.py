import json
import pdb
import codecs
import numpy as np
import pandas as pd

def get_data():
    tweets = []
    df_Chinese = pd.read_csv('train_set.csv')
    total = df_Chinese['parsedText'].count()
    df_Chinese['label'] = df_Chinese['label'].astype(np.int64, errors = 'ignore')
    df_Chinese['parsedText'] = df_Chinese['parsedText'].astype(str)
    for num in range(0,total):
        #print(line)
        tweets.append({
            'text': df_Chinese['parsedText'][num],
            'label': df_Chinese['label'][num]
        })
    #print(tweets[0])
    #pdb.set_trace()
    return tweets


if __name__=="__main__":
    tweets = get_data()
    pdb.set_trace()
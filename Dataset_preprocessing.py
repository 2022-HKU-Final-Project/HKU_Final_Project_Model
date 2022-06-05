#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import read_parquet
import matplotlib.pylab as plt
import seaborn as sns
import re
from string import punctuation
from lxml import etree
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def read_dataset(data):
    if data.endswith(".csv"):
        df = pd.read_csv(data)
    else:
        df = read_parquet(data)
    return df

def preprocess_column(df):
    # NaN 
    values = {"jobDiploma": "无", "jobPosition": "无", "jobCate": "无", "jobDesc": "无", "jobSalary_format": 0, 
         "jobWorkAge": "经验不限", "people_count": "其它"}
    df = df.fillna(value = values)
    # jobDiploma
    df.loc[df.jobDiploma == '2', 'jobDiploma'] = '学历不限'
    df.loc[df.jobDiploma == '3', 'jobDiploma'] = '学历不限'
    df.loc[df.jobDiploma == '5', 'jobDiploma'] = '学历不限'
    df.loc[df.jobDiploma == '10', 'jobDiploma'] = '大专'
    df.loc[df.jobDiploma == '不限', 'jobDiploma'] = '学历不限'
    df.loc[df.jobDiploma == '中专', 'jobDiploma'] = '中专/中技'
    df.loc[df.jobDiploma == '中技', 'jobDiploma'] = '中专/中技'
    df.loc[df.jobDiploma == '中专', 'jobDiploma'] = '中专/中技'
    df.loc[df.jobDiploma == '技校', 'jobDiploma'] = '中专/中技'
    df.loc[df.jobDiploma == '学历不限', 'Diploma'] = 0
    df.loc[df.jobDiploma == '初中及以下', 'Diploma'] = 1
    df.loc[df.jobDiploma == '中专及以下', 'Diploma'] = 2
    df.loc[df.jobDiploma == '中专/中技', 'Diploma'] = 3
    df.loc[df.jobDiploma == '高中', 'Diploma'] = 4
    df.loc[df.jobDiploma == '大专', 'Diploma'] = 5
    df.loc[df.jobDiploma == '本科', 'Diploma'] = 6
    df.loc[df.jobDiploma == '硕士', 'Diploma'] = 7
    df.loc[df.jobDiploma == '博士', 'Diploma'] = 8
    df.loc[df.jobDiploma == '无', 'Diploma'] = 9
    
    # jobWorkAge
    df.loc[df.jobWorkAge == '招10人', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '招若干人', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '10年以上', 'jobWorkAge'] = '10年以上经验'
    df.loc[df.jobWorkAge == '1年以下', 'jobWorkAge'] = '1年经验'
    df.loc[df.jobWorkAge == '1年以内', 'jobWorkAge'] = '1年经验'
    df.loc[df.jobWorkAge == '不限', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '无经验', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '经验不限,可接收应届生', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '1-2年,可接收应届生', 'jobWorkAge'] = '1-2年'
    df.loc[df.jobWorkAge == '应届毕业生', 'jobWorkAge'] = '应届生'
    df.loc[df.jobWorkAge == '招2人', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '招3人', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '招人', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '无工作经验', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '8-9年经验', 'jobWorkAge'] = '8-10年'
    df.loc[df.jobWorkAge == '2年经验', 'jobWorkAge'] = '1-2年'
    df.loc[df.jobWorkAge == '大专', 'jobWorkAge'] = '经验不限'
    df.loc[df.jobWorkAge == '经验不限', 'WorkAge'] = 0
    df.loc[df.jobWorkAge == '1-3年', 'WorkAge'] = 1
    df.loc[df.jobWorkAge == '1年经验', 'WorkAge'] = 2
    df.loc[df.jobWorkAge == '1-2年', 'WorkAge'] = 3
    df.loc[df.jobWorkAge == '3-5年', 'WorkAge'] = 4
    df.loc[df.jobWorkAge == '3-4年经验', 'WorkAge'] = 5
    df.loc[df.jobWorkAge == '应届生', 'WorkAge'] = 6
    df.loc[df.jobWorkAge == '5-10年', 'WorkAge'] = 7
    df.loc[df.jobWorkAge == '5-7年经验', 'WorkAge'] = 8
    df.loc[df.jobWorkAge == '10年以上经验', 'WorkAge'] = 9
    df.loc[df.jobWorkAge == '6-7年', 'WorkAge'] = 10
    df.loc[df.jobWorkAge == '1年以下,可接收应届生', 'WorkAge'] = 11
    df.loc[df.jobWorkAge == '8-10年', 'WorkAge'] = 12
    df.loc[df.jobWorkAge == '3-5年,可接收应届生', 'WorkAge'] = 13
    df.loc[df.jobWorkAge == '6-7年,可接收应届生', 'WorkAge'] = 14
    
    # people count
    df.loc[df.people_count == '1-49人', 'people_count'] = '1-49'
    df.loc[df.people_count == '50-99人', 'people_count'] = '50-99'
    df.loc[df.people_count == '100-499人', 'people_count'] = '100-499'
    df.loc[df.people_count == '500-999人', 'people_count'] = '500-999'
    df.loc[df.people_count == '10000人以上', 'people_count'] = '10000+'
    df.loc[df.people_count == '1000-9999人', 'people_count'] = '1000-9999'
    df.loc[df.people_count == '其它', 'people_count'] = 'others'
    
    df.loc[df.people_count == '1-49', 'people'] = 0
    df.loc[df.people_count == '50-99', 'people'] = 1
    df.loc[df.people_count == '100-499', 'people'] = 2
    df.loc[df.people_count == '500-999', 'people'] = 3
    df.loc[df.people_count == '10000+', 'people'] = 4
    df.loc[df.people_count == '1000-9999', 'people'] = 5
    df.loc[df.people_count == 'others', 'people'] = 6
    
    return df

def preprocess_Chinese(text_string):
    """
    Accepts a text string and replaces:
    1) urls with <URL>
    2) lots of whitespace with one instance
    3) mentions with user
    4) hashtags with contents of hashtags without #
    5) punctuation symbols
    6) Special symbols
    7) time
    8) traditional Chinese characters with simplified Chinese characters
    9) date
    10) RT
    11) HTML tags
    12) title symbol

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned

    Returns parsed text.
    """

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    #hashtag_regex = '#[\w\-]+'
    titleSymbol_regex = '([A-Z]+(\.[A-Z]+)?|[0-9]+(\.[0-9]+)?)(\u3001|\.|\))'
    rt_regex = '\\b[Rr][Tt]\\b'
    punc_regex = punctuation + u'《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    punc = u'_つ____つっ__ㄟφㄇㄟㄉωㄍㄌДˇㄌㄅ一ㄝづσさえちオオグソクムシㄏㄚㄨか'
    punc2 = u'_'
    month_regex = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    #week_regex = u'Mon Tues Wed Thur Fri Sat Sun'
    parsed_text = re.sub(giant_url_regex, '<url>', text_string)
    parsed_text = re.sub(titleSymbol_regex, "", parsed_text)
    #parsed_text = re.sub(titleSymbol_regex2, "", parsed_text)
    parsed_text = etree.HTML(parsed_text)  # 解析html
    parsed_text = parsed_text.xpath('string(.)')
    parsed_text = re.sub(mention_regex, 'user', parsed_text)
    #parsed_text = re.sub(titleSymbol_regex, "", parsed_text)
    #parsed_text = emoji.demojize(parsed_text)
    #parsed_text = tradition2simple(parsed_text)
    #parsed_text = re.sub(hashtag_regex, '', parsed_text)
    parsed_text = re.sub(r"(\d{1,2}/\d{1,2}\s\d{1,2}:\d{1,2})", "", parsed_text)  # 去除时间 xx/xx xx:xx
    parsed_text = re.sub(r"(\d{1,2}:\d{1,2}:\d{1,2}\s\d{4})", "", parsed_text)  # 去除时间 xx:xx:xx xxxx
    parsed_text = re.sub(r'[^\w\s]', "", parsed_text)  # 去除标点
    parsed_text = re.sub(r"[{}]+".format(punc2), " ", parsed_text)  # 去除特殊符号
    parsed_text = re.sub(r"[{}]+".format(punc), "", parsed_text)  # 去除特殊符号
    parsed_text = re.sub(rt_regex, '', parsed_text)
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = parsed_text.replace('\n', '')
    parsed_text = parsed_text.strip()
    parsed_text = parsed_text.lower()
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    #print(parsed_text)
    return parsed_text

def split_sent(text):
    stopwords = []
    with open('stopwords-master/cn_stopwords.txt', 'r') as f:
        for eachline in f.readlines():
            eachline = eachline.replace("\n", "")
            stopwords.append(eachline)
        f.close()
    # 分词
    """将sent切分成tokens"""
    tokens = []
    sent = text.strip()#去除首尾的空格
    for token in jieba.cut(sent):
        if token not in stopwords and token != ' ':
            tokens.append(token)    
    return tokens


# In[2]:


data = "xxxx.csv" # dataset path
df = read_dataset(data)
df = preprocess_column(df)

parsed_text = []
for sen in df['jobDesc'][0:]:
    sentence = preprocess_Chinese(sen)
    parsed_text.append(sentence.strip())# 去除收尾空格后，加入list
df['parsedText'] = parsed_text

split_sentence = []
for i in parsed_text:
    split_sentence.append(split_sent(i))
df['split_word'] = split_sentence

y_df = df.iloc[:, 0]
y = LabelEncoder().fit_transform(y_df.values)
df['label'] = y
data_train, data_test = train_test_split(df, test_size=0.2, random_state=1234, stratify = df['label'])

data_train.to_csv("train_set.csv")
data_test.to_csv("test_set.csv")


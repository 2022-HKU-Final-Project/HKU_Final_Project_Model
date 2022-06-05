import sys
import re
from string import punctuation
import emoji
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
    8) emoji/smileys with specific meanings of emoji
    9) traditional Chinese characters with simplified Chinese characters
    10) date
    11) RT

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned

    Returns parsed text.
    """

    '''space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    #hashtag_regex = '#[\w\-]+'
    rt_regex = '\\b[Rr][Tt]\\b'
    punc_regex = punctuation + u'《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    punc = u'_つ____つっ__ㄟφㄇㄟㄉωㄍㄌДˇㄌㄅ一ㄝづσさえちオオグソクムシㄏㄚㄨか'
    punc2 = u'_'
    month_regex = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    #week_regex = u'Mon Tues Wed Thur Fri Sat Sun'
    parsed_text = re.sub(giant_url_regex, '<url>', text_string)
    parsed_text = re.sub(mention_regex, 'user', parsed_text)
    parsed_text = emoji.demojize(parsed_text)
    #parsed_text = tradition2simple(parsed_text)
    #parsed_text = re.sub(hashtag_regex, '', parsed_text)
    parsed_text = re.sub(r"(\d{1,2}/\d{1,2}\s\d{1,2}:\d{1,2})", "", parsed_text)  # 去除时间 xx/xx xx:xx
    parsed_text = re.sub(r"(\d{1,2}:\d{1,2}:\d{1,2}\s\d{4})", "", parsed_text)  # 去除时间 xx:xx:xx xxxx
    parsed_text = re.sub(r'[^\w\s]', "", parsed_text)  # 去除标点
    parsed_text = re.sub(r"[{}]+".format(punc2), " ", parsed_text)  # 去除特殊符号
    parsed_text = re.sub(r"[{}]+".format(punc), "", parsed_text)  # 去除特殊符号
    parsed_text = re.sub(rt_regex, '', parsed_text)
    #parsed_text = re.sub(r"[]+".format(month_regex), "", parsed_text)#月份
    #parsed_text = re.sub(r"[{}]+".format(week_regex), "", parsed_text)#星期
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = parsed_text.replace('\n', '')
    parsed_text = parsed_text.strip()
    parsed_text = parsed_text.lower()
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    #print(parsed_text)'''
    # We have preprocessed text before
    return text_string


if __name__ == '__main__':
    _, text = sys.argv
    if text == "test":
        text = u"我爱你#HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    tokens = preprocess_Chinese(text)
    print(tokens)

from string import punctuation
import jieba
#jieba.load_userdict("../../../../data/sensitive words.txt")
#from preprocess_twitter import preprocess_Chinese as tokenizer_g

def glove_tokenize(text):
    '''text = tokenizer_g(text)
    word = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ' ', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    tokens = []
    text = text.strip()#去除首尾的空格
    for token in jieba.cut(text):
        m = 0
        for i in word:
            #print(i)
            if token == i or token.isdigit():
                break
            else:
                m += 1
                if (m <= 19):
                    continue
                else:
                    tokens.append(token)'''
    stopwords = []
    with open('cn_stopwords.txt', 'r') as f:
        for eachline in f.readlines():
            eachline = eachline.replace("\n", "")
            stopwords.append(eachline)
        f.close()
    tokens = []
    sent = text.strip()
    for token in jieba.cut(sent):
        if token not in stopwords and token != ' ':
            tokens.append(token)
    print(tokens)
    return tokens

str = ['我爱你 就这么简单', 'me too']
glove_tokenize(str[0])
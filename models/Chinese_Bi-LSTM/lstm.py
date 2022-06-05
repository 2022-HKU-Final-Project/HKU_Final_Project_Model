from data_handler import get_data
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
import codecs
import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
from batch_gen import batch_gen
import sys

from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize

#from keras_contrib.layers.crf import CRF
#from keras_contrib.utils import save_load_utils

### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}



EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 64
SCALE_LOSS_FUN = None

word2vec_model = None



def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception as e:
        print('Encoding not found: %s' %(word))
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    # 词向量中没有的词
    print("%d embedding missed"%n)
    return embedding


def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = TOKENIZER(tweet['text'])
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
        else:
            print('huhuhahei')
            print(tweet)
    print('Tweets selected:', len(tweet_return))
    #pdb.set_trace()
    return tweet_return


def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet in tweets:
        words = TOKENIZER(tweet['text'])
        print(words)
        #text = ''.join([c for c in text if c not in punctuation])
        #words = text.split()
        #words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'


def filter_vocab(k):
    global freq, vocab
    pdb.set_trace()
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence():
    #y_map = {
            #'none': 0,
            #'sexism': 1
            #}

    X, y = [], []
    for tweet in tweets:
        #print('come on !' + tweet)
        #text = TOKENIZER(tweet['text'])
        words = TOKENIZER(tweet['text'])
        #print(text)
        #text = ''.join([c for c in text if c not in punctuation])
        #words = text.split()
        #words = [word for word in words if word not in STOPWORDS]
        #print(words)
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
            #print(seq)
        X.append(seq)
        y.append(tweet['label'])
    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'Bi-LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    #model.add(AttentionLayer(attention_size = 50))
    model.add(Dense(65, activation='softmax'))
    #crf_layer = CRF(2)
    #model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])

    print(model.summary())
    return model


def train_LSTM(X, y, model, inp_dim, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train)) #矩阵进行 行连接
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]
                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))

                try:
                    y_temp = np_utils.to_categorical(y_temp, num_classes=65)
                except Exception as e:
                    print(e)
                    print(y_temp)
                print(x.shape, y.shape)
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                print(loss, acc)

        y_pred = model.predict_on_batch(X_test)
        y_pred2 = model.predict_proba(X_test)[0,1]
        print(y_pred2[0:5])
        y_pred = np.argmax(y_pred, axis=1)
        #print(softmax(y_pred[0:5]))
        print(classification_report(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred))
        print(y_pred)
        #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,
                                                                        #model.predict_proba(X_test)[:, 1])
        #roc_auc = auc(false_positive_rate, true_positive_rate)  # 求面积
        #print(roc_auc)

        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')


    print("macro results are")
    print("average precision is %f" %(p/NO_OF_FOLDS))
    print("average recall is %f" %(r/NO_OF_FOLDS))
    print("average f1 is %f" %(f1/NO_OF_FOLDS))

    print("micro results are")
    print("average precision is %f" %(p1/NO_OF_FOLDS))
    print("average recall is %f" %(r1/NO_OF_FOLDS))
    print("average f1 is %f" %(f11/NO_OF_FOLDS))
    model.save('saved_bilstm.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)


    args = parser.parse_args()
    GLOVE_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    KERNEL = args.kernel
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function



    np.random.seed(SEED)
    print('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print('Embedding Dimension: %d' %(EMBEDDING_DIM))
    print('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)

    tweets = select_tweets()
    gen_vocab()
    #filter_vocab(20000)
    X, y = gen_sequence()
    print(X[0:10])
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()

    model = lstm_model(data.shape[1], EMBEDDING_DIM)
    #model = lstm_model(data.shape[1], 25, get_embedding_weights())
    train_LSTM(data, y, model, EMBEDDING_DIM, W)

    pdb.set_trace()
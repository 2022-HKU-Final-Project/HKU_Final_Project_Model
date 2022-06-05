from gensim.scripts.glove2word2vec import glove2word2vec


def transfer(gloveFile, word2vecFile):
    glove2word2vec(gloveFile, word2vecFile)


if __name__ == '__main__':
	## the filepath is the path of your glove document
    transfer('/Users/tanwenting/Desktop/毕业设计/glove.twitter.27B/glove.twitter.27B.25d.txt', '/Users/tanwenting/Desktop/毕业设计/glove_twitter_27B_25d_w2v_format.txt')
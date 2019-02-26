import operator
from tqdm import tqdm


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')


def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


glove_path = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
embeddings_ind = dict(get_coefs(*o.split(" ")) for o in open(glove_path))
train_df = pd.read_csv("../input/train.csv")
print("Train shape : {}".format(train_df.shape))

sentences = train_df["question_text"].apply(lambda x: x.split()).values
vocab = build_vocab(sentences)

oov = check_coverage(vocab, embeddings_ind)
print(oov[:20])




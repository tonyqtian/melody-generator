import collections
import numpy as np
import re


class TextProcessor(object):
    @staticmethod
    def from_file(input_file):
        with open(input_file, 'r', encoding = 'utf8') as fh:
            text = fh.read()
        return TextProcessor(text)

    def __init__(self, text):
#         self.words = self._text2words(text)
        self.words = [w for w in text.split()]
        self.id2word = None
        self.word2id = None
        self.vector = None

    def set_vocab(self, word2id):
        self.word2id = word2id
        return self

    def create_vocab(self, size):
        counter = collections.Counter(self.words)
        print( 'Vocabulary size reduced from %s to %s' % (len(counter), size) )
        count_pairs = counter.most_common(size-1)
        self.id2word = list(dict(count_pairs).keys())
        self.id2word[-1] = '<unk>'
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def get_vector(self):
        unk = self.word2id['<unk>']
        self.vector = [self.word2id[word] if word in self.word2id else unk for word in self.words]
        return self.vector

    def save_converted(self, filename):
        with open(filename, 'w') as fh:
            for wid in self.vector:
                fh.write(self.id2word[wid]+' ')

    @staticmethod
    def _text2words(text):
        # prepare for word based processing
        re4 = re.compile(r'\.\.+')
        re5 = re.compile(r' +')

        text = text.lower()
        text = re4.sub(' <3dot> ', text)
        text = text.replace(',', ' , ')
        text = text.replace('.', ' . ')
        text = text.replace('/', ' . ')
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('[', ' ( ')
        text = text.replace(']', ' ) ')
        text = text.replace(':', ' : ')
        text = text.replace("'", " '")
        text = text.replace('?', ' ? ')
        text = text.replace(';', ' . ')
        text = text.replace('-', ' -')

        text = text.replace('<3dot>', ' ... ')
        text = text.replace('"', '')

        text = re5.sub(' ', text)
        text = text.replace('\n', ' <nl> ')
        return ['\n' if w == '<nl>' else w for w in text.split()]


def train_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

import config
import pandas as pd
import numpy as np
import random
from utilities import format_string
import os

labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


# def get_train_dev_corpus_file_name(label):
#     return '{}/text/{}/train-corpus-{}.txt'.format(config.root, label), \
#            '{}/text/{}/dev-corpus-{}.txt'.format(config.root, label)

def get_lable_corpus_file_name(label):
    dir = '{}/text/{}'.format(config.root, label)
    if not os.path.exists(dir):
        os.mkdir(dir)

    return '{}/positive.txt'.format(dir), '{}/negative.txt'.format(label)


def write_one_train_corpus(label):
    original_content = pd.read_csv('data/train.csv')

    labeled_data = original_content[label].tolist()
    black_indices = np.nonzero(np.array(labeled_data) == 1)[0].tolist()
    white_indices = np.nonzero(np.array(labeled_data) == 0)[0].tolist()

    major_length = len(white_indices)

    expanding_ratio = len(white_indices) // len(black_indices) + 1
    black_indices = black_indices * expanding_ratio
    print('expanding ratio is {}'.format(expanding_ratio))
    #
    length = min(len(white_indices), len(black_indices))

    white_indices = white_indices[:length]
    black_indices = black_indices[:length]
    #
    print('white indices is {}'.format(len(white_indices)))
    print('black indices is {}'.format(len(black_indices)))
    #
    indices = black_indices + white_indices

    assert len(black_indices) == len(white_indices) == major_length

    [random.shuffle(indices) for _ in range(10)]

    # train_ratio = 0.75
    # train_length = int(len(indices) * train_ratio)
    # train_indices = indices[: train_length]
    # dev_indices = indices[train_length:]

    pos_file, neg_file = get_lable_corpus_file_name(label)
    sentences = original_content['comment_text'].tolist()
    labels = original_content[label].tolist()

    assert len(sentences) == len(labels)

    def write_to_file(file, indices):
        with open(file, 'w', encoding='utf-8') as f:
            for ii in indices:
                sentence = format_string(sentences[ii])
                f.write('{}\n'.format(sentence))

    write_to_file(pos_file, black_indices)
    write_to_file(neg_file, white_indices)


if __name__ == '__main__':
    for l in labels:
        print('label {}'.format(l))
        write_one_train_corpus(l)

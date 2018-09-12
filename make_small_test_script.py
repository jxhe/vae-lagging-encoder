import torch
import numpy as np

def select_indices_yahoo():
    f = open('yahoo_data/yahoo_10percent_indices.test', 'w')
    selected = np.random.choice(range(10000), 1000, replace=False)
    print(selected)
    for sel in selected:
        f.write(str(sel)+'\n')

def load_indices_yahoo():
    f = open('yahoo_data/yahoo_10percent_indices.test', 'r')
    indices = []
    for indice in f:
        indices.append(int(indice))
    return indices


def select_test_samples_yahoo(test_indices):
    test_data = open('yahoo_data/yahoo.test.txt', 'r')
    small_test_data = open('yahoo_data/yahoo_10percent.test.txt', 'w')
    for i, line in enumerate(test_data):
        if i in test_indices:
            small_test_data.write(line)


def select_indices_omniglot():
    f = open('omniglot_data/omniglot_small_percent_indices.test', 'w')
    selected = np.random.choice(range(8070), 1000, replace=False)
    print(selected)
    for sel in selected:
        f.write(str(sel)+'\n')
    pass

def load_indices_omniglot():
    f = open('omniglot_data/omniglot_small_percent_indices.test', 'r')
    indices = []
    for indice in f:
        indices.append(int(indice))
    return indices



def select_indices_yelp():
    f = open('yelp_data/yelp_10percent_indices.test', 'w')
    selected = np.random.choice(range(10000), 1000, replace=False)
    print(selected)
    for sel in selected:
        f.write(str(sel)+'\n')

def load_indices_yelp():
    f = open('yelp_data/yelp_10percent_indices.test', 'r')
    indices = []
    for indice in f:
        indices.append(int(indice))
    return indices


def select_test_samples_yelp(test_indices):
    test_data = open('yelp_data/yelp.test.txt', 'r')
    small_test_data = open('yelp_data/yelp_10percent.test.txt', 'w')
    for i, line in enumerate(test_data):
        if i in test_indices:
            small_test_data.write(line)


if __name__ == '__main__':
    #omniglot
    # select_indices_omniglot()
    # test_indices = load_indices_omniglot()

    #yahoo
    # select_indices_yahoo()
    # test_indices = load_indices_yahoo()
    # select_test_samples_yahoo(test_indices)
    ######
    #yelp
    # select_indices_yelp()
    # test_indices = load_indices_yelp()
    # select_test_samples_yelp(test_indices)
    ######
    pass
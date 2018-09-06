import numpy as np

def select_indices():
    f = open('yahoo_data/yahoo_10percent_indices.test', 'w')
    selected = np.random.choice(range(10000), 1000, replace=False)
    print(selected)
    for sel in selected:
        f.write(str(sel)+'\n')

def load_indices():
    f = open('yahoo_data/yahoo_10percent_indices.test', 'r')
    indices = []
    for indice in f:
        indices.append(int(indice))
    return indices


def select_test_samples(test_indices):
    test_data = open('yahoo_data/yahoo.test.txt', 'r')
    small_test_data = open('yahoo_data/yahoo_10percent.test.txt', 'w')
    for i, line in enumerate(test_data):
        if i in test_indices:
            small_test_data.write(line)

if __name__ == '__main__':
    # select_indices()
    # test_indices = load_indices()
    # select_test_samples(test_indices)
    pass
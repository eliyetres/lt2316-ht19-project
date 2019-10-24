import pickle


def load_pickle(filename):
    """ Loads a pickled file """
    pickle_load = pickle.load(open(filename, 'rb'))
    return pickle_load

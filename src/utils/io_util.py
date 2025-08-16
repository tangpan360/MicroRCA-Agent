import pickle


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
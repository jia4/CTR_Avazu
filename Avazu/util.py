from csv import DictReader
from math import exp, log, sqrt

'''
function readData is to read the file using csv DictReader
the D is dimension, which is the range of the weight list
'''


def readData(path, D):
    for row in DictReader(open(path)):
        ID = row['id']
        del row['id']

        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        row['hour'] = row['hour'][6:]

        x = []
        for key in row:
            value = row[key]
            # one-hot encode with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield ID,x, y

def log_loss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def readData_ffm(path, D):
    for row in DictReader(open(path)):
        ID = row['id']
        del row['id']

        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        row['hour'] = row['hour'][6:]

        x = []
        for key in row:
            value = row[key]
            # one-hot encode with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append((index,key))

        yield ID,x, y
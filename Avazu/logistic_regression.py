from math import exp, log, sqrt


class logistic_regression(object):
    def __init__(self, alpha, L2, D, method):
        self.alpha = alpha
        self.L2 = L2

        self.D = D
        self.method = method

        self.w = [0.5] * D

    def _indices(self, x):
        method = self.method
        yield 0

        for index in x:
            yield index

        if method == 'interaction':
            D = self.D
            L = len(x)
            x = sorted(x)
            for i in range(L):
                for j in range(i + 1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        w = self.w

        # h is the h(wx) in sigmoid function
        h = 0

        for i in self._indices(x):
            h += w[i]

        return 1. / (1. + exp(-h))

    def update(self, x, p, y):
        # p is the prediction proba and y is the label
        g = p - y

        alpha = self.alpha
        L2 = self.L2
        w = self.w

        for i in self._indices(x):
            w[i] -= alpha * (g + L2 * w[i])

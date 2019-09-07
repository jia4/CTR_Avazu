from math import exp, log, sqrt

class fm_regression(object):
    def __init__(self, alpha, L2, D):
        self.alpha = alpha
        self.L2 = L2

        self.D = D

        self.w = [0.5] * D

    def _indices(self, x):
        yield 0

        for index in x:
            yield index


    def predict(self, x):
        w = self.w
        D = self.D

        # h is the h(wx) in sigmoid function
        h = 0

        for i in self._indices(x):
            h += w[i]

        L = len(x)
        self.sum_1 = 0
        self.sum_2 = 0
        self.sum_3 = 0
        for i in range(L):
            i_1 = abs(hash(str(x[i]) + '_' + '1'))%D
            self.sum_1 += w[i_1]
            i_2 = abs(hash(str(x[i]) + '_' + '2'))%D
            self.sum_2 += w[i_2]
            i_3 = abs(hash(str(x[i]) + '_' + '3'))%D
            self.sum_3 += w[i_3]
            for j in range(i+1,L):
                j_1 = abs(hash(str(x[j]) + '_' + '1'))%D
                j_2 = abs(hash(str(x[j]) + '_' + '2'))%D
                j_3 = abs(hash(str(x[j]) + '_' + '3'))%D
                h += w[i_1] * w[j_1]
                h += w[i_2] * w[j_2]
                h += w[i_3] * w[j_3]

        return 1. / (1. + exp(-h))

    def update(self, x, p, y):
        D = self.D
        # p is the prediction proba and y is the label
        g = p - y

        alpha = self.alpha
        L2 = self.L2
        w = self.w

        for i in self._indices(x):
            w[i] -= alpha * (g + L2 * w[i])

        for i in range(len(x)):
            i_1 = abs(hash(str(x[i]) + '_' + '1'))%D
            w[i_1] -= alpha*(g*(self.sum_1-w[i_1]) + L2 * w[i_1])
            i_2 = abs(hash(str(x[i]) + '_' + '2')) % D
            w[i_2] -= alpha * (g * (self.sum_2 - w[i_2]) + L2 * w[i_2])
            i_3 = abs(hash(str(x[i]) + '_' + '3')) % D
            w[i_3] -= alpha * (g * (self.sum_3 - w[i_3]) + L2 * w[i_3])
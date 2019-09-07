from math import exp, log, sqrt

class ffm_regression(object):
    def __init__(self, alpha, L2, D):
        self.alpha = alpha
        self.L2 = L2

        self.D = D

        self.w = [0.5] * D

    def _indices(self, x):
        yield 0

        for index in x:
            yield index[0]


    def predict(self, x):
        w = self.w
        D = self.D

        # h is the h(wx) in sigmoid function
        h = 0

        for i in self._indices(x):
            h += w[i]

        L = len(x)
        for i in range(L):
            for j in range(i+1,L):
                i_1 = abs(hash(str(x[i][0]) + '_' + '1'+'_'+x[j][1]))%D
                i_2 = abs(hash(str(x[i][0]) + '_' + '2'+'_'+x[j][1]))%D
                i_3 = abs(hash(str(x[i][0]) + '_' + '3'+'_'+x[j][1]))%D

                j_1 = abs(hash(str(x[j][0]) + '_' + '1'+'_'+x[i][1]))%D
                j_2 = abs(hash(str(x[j][0]) + '_' + '2'+'_'+x[i][1]))%D
                j_3 = abs(hash(str(x[j][0]) + '_' + '3'+'_'+x[i][1]))%D

                h += w[i_1] * w[j_1]
                h += w[i_2] * w[j_2]
                h += w[i_3] * w[j_3]

        return 1. / (1. + exp(-max(min(h,35.),-35.)))

    def update(self, x, p, y):
        D = self.D
        # p is the prediction proba and y is the label
        g = p - y

        alpha = self.alpha
        L2 = self.L2
        w = self.w

        for i in self._indices(x):
            w[i] -= alpha * (g + L2 * w[i])

        # update by every field-hyper vector
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                i_1 = abs(hash(str(x[i][0]) + '_' + '1'+'_'+x[j][1]))%D
                j_1 = abs(hash(str(x[j][0]) + '_' + '1' + '_' + x[i][1])) % D
                w[i_1] -= alpha*(g*w[j_1]+L2*w[i_1])
                w[j_1] -= alpha * (g * w[i_1] + L2 * w[j_1])

                i_2 = abs(hash(str(x[i][0]) + '_' + '2' + '_' + x[j][1])) % D
                j_2 = abs(hash(str(x[j][0]) + '_' + '2' + '_' + x[i][1])) % D
                w[i_2] -= alpha * (g * w[j_2] + L2 * w[i_2])
                w[j_2] -= alpha * (g * w[i_2] + L2 * w[j_2])

                i_3 = abs(hash(str(x[i][0]) + '_' + '3' + '_' + x[j][1])) % D
                j_3 = abs(hash(str(x[j][0]) + '_' + '3' + '_' + x[i][1])) % D
                w[i_3] -= alpha * (g * w[j_3] + L2 * w[i_3])
                w[j_3] -= alpha * (g * w[i_3] + L2 * w[j_3])


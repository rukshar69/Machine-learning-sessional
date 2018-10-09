import bisect

class Weights:

    weights = []
    def __init__(self,size):
        self.dummy = 0;
        self.size = size;
        self.initialize()

    def initialize(self):
        frequency = 1 / self.size
        self.weights = [frequency]*self.size

    def normalize(self):
        w = self.weights
        sum_ = sum(w);
        w = [i/sum_ for i in w]
        self.weights = w;

    def sum_weights(self,weights): #cumulation
        len_w8 = len(weights)
        sum_w8 = [sum(weights[0:x + 1]) for x in range(0, len_w8)]
        return sum_w8

    def index_sum_weight(self,sum_w8, upper_limit):
        index = bisect.bisect(sum_w8,upper_limit)
        return index



'''

w = Weights(5)

print(len(w.weights))
print(w.weights)
#w.weights[3] = .5;
#print(w.weights)
#w.normalize()
#print(w.weights)
w_ = w.sum_weights(w.weights)
print(w_)
print(w.index_sum_weight(w_,.9))
'''


from  ReadCSV import  ReadCSV
from  Weights import Weights
from  DecisionTree import Stump
import math
from Adaboost import Adaboost

class FScore:
    '''
    real = yes and pred = yes truepos
    real =no and pred = yes falsepos
    real = no and pred =no trueNeg
    real = yes and pred = no falseNeg
    '''

    def __init__(self,db, learner, learner_priority):
        self.dummy = 0;
        self.db = db
        self.learner = learner
        self.learner_priority = learner_priority
        self.pos_neg = [0,0,0,0]

    def test(self,r):

        pred_values = []
        for i in range(len(self.learner)):
            pred_values.append(self.learner[i].decide_for_test(r))
        #print(pred_values)
        #print(self.learner_priority)
        yes_point = [];
        no_point = [];
        for i in range(len(self.learner)):
            t = pred_values[i]
            if t == 'yes':
                yes_point +=  [self.learner_priority[i]]
            else :
                no_point += [self.learner_priority[i]]


        yes_point = sum(yes_point)
        no_point = sum(no_point)

        dec = 'yes' if yes_point>no_point else 'no'
        return dec

    def determine_posNeg(self,pred_values, real_values):
        length = len(pred_values)
        for i in range(length):
            real = real_values[i]
            pred = pred_values[i]
            if real =='yes':
                if pred == 'yes':
                    self.pos_neg[0] = self.pos_neg[0] +1
                elif pred == 'no':
                    self.pos_neg[3] = self.pos_neg[3] + 1
            elif real == 'no':
                if pred == 'yes':
                    self.pos_neg[1] = self.pos_neg[1] +1
                elif pred == 'no':
                    self.pos_neg[2] = self.pos_neg[2] + 1

    def det_precision(self):

        sum_ = self.pos_neg[0]+self.pos_neg[1]
        precision= 0 if sum_ == 0 else self.pos_neg[0]/sum_
        return  precision

    def det_recall(self):

        sum_ = self.pos_neg[0] + self.pos_neg[3]
        recall = 0 if sum_ == 0 else self.pos_neg[0]/sum_
        return recall
    def fscore(self,p,r):
        num = 2*p*r
        denom = p+r
        fscr = num/denom
        return fscr

    def det_accuracy(self):
        numCorrect = self.pos_neg[0]+self.pos_neg[2]
        dem = len(self.db)-1
        accuracy = numCorrect/dem
        return accuracy

    def determine_fscore(self):
        pred_values = []
        real_values = []
        len_db = len(self.db)
        for r in self.db[1:]:
            pred_values.append(self.test(r))
            real_values.append(r[20])
        #print(pred_values)
        #print(real_values)

        self.determine_posNeg( pred_values, real_values)
        #print(self.pos_neg)
        p = self.det_precision()
        r = self.det_recall()
        f = self.fscore(p,r)
        #print(p)
        #print(f)
        accuracy = self.det_accuracy();


        return f,accuracy






#############################################################################

'''
rd = ReadCSV();
db_1= rd.produceDB()
ada = Adaboost(db_1,5)
ada.algo();
learner_ = ada.learners
learn_prio =ada.learner_priority
#for l in learner:
 #   l.print_decision()

f =FScore(db_1[:10],learner_,learn_prio)
#fscr = f.determine_fscore()
#print(fscr)
'''
'''
db_ = db_1[:5]
for r in db_:
    d = f.test(r)
    print(d)
'''
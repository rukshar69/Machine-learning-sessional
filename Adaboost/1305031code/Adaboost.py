

from  ReadCSV import  ReadCSV
from  Weights import Weights
from  DecisionTree import Stump
import math

class Adaboost:
    db = []
    learners = []
    learner_priority=[]
    number_learners = 0
    sample_limit = 20

    def __init__(self,db,learner_no):
        self.dummy = 0;
        self.db = db;
        self.w = Weights(len(db)-1)
        self.number_learners = learner_no
        self.learners = []
        self.learner_priority = []

    def createSample(self):
        sum_w8 = self.w.sum_weights(self.w.weights)
        #print(sum_w8)
        rd = ReadCSV();
        sample = rd.create_sampleList(self.sample_limit, self.db, sum_w8)
        return  sample
    def create_learner(self,sample):
        s = Stump(sample);
        s.tree();
        #s.print_decision()
        self.learners.append(s)

    def determine_error(self,index_learner):
        rr = 0
        db_ = self.db
        dec_values = []
        pred_values = []
        for j in range(1, len(db_)):
            temp = self.db[j]
            # print("val of data ",temp)
            tval = self.learners[index_learner].decide_for_test(temp)
            # print(tval)
            pred_values.append(tval)
            actual_dec = temp[20]
            dec_values.append(actual_dec)
            if tval != actual_dec:
                rr += self.w.weights[j-1]
        return rr, dec_values,pred_values
    def modify_weights(self,dec_val,pred_val,rr):
        db_ = self.db

        for j in range(1, len(db_)):
            if pred_val[j - 1] == dec_val[j - 1]:
                self.w.weights[j-1] *= rr;

    def updatePriority(self,rr):
        self.w.normalize()
        # print(w.weights)
        rr = 1 / rr
        rr = math.log(rr, 2)
        #print(rr)
        self.learner_priority.append(rr)

    def algo(self):

        for i in range(self.number_learners):
            #print("learner no ",k)
            #print(self.w.weights)
            sample = self.createSample()
            #for r in sample:
             #   print(r)
            self.create_learner(sample)

            rr,dec_val,pred_val=self.determine_error(i)
            #print(rr)
            #print(len(dec_val))
            rr = rr / (1 - rr)
            self.modify_weights(dec_val,pred_val,rr)

            self.updatePriority(rr)

        return self.learners, self.learner_priority




        #for l in self.learners:
        #   l.print_decision()




##########################################################################################

'''
def printList(list_):
    for r in list_:
        print(r)
rd = ReadCSV();
db_1= rd.produceDB()
#print(db_1[23])
w = Weights(len(db_1))
w8 = w.weights;
#print(w.sum_weights(w8))
w_sum = w.sum_weights(w8)
samp = rd.create_sampleList(20,db_1,w_sum)
#for r in samp:
#    print(r)

#s = Stump(samp);
#s.tree();
#print(len(types))
#printList(a)
#print(len(attrName))
'''
'''
rd = ReadCSV();
db_1= rd.produceDB()
ada = Adaboost(db_1,5)
ada.algo();
'''

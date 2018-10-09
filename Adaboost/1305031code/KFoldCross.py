

from  ReadCSV import  ReadCSV
from  Weights import Weights
from  DecisionTree import Stump
import math
from Adaboost import Adaboost
from FScore import  FScore

class KFoldCross:
    def __init__(self,db,turns):
        self.dummy = 0;
        self.db =db
        self.turns  = turns
        self.attr_list = db[0]
        self.block_size = int((len(db)-1) /turns)

    def divided_db(self,db_):
        div_db = []
        for i in range(0,len(db_),self.block_size):
            temp = db_[i:i+self.block_size]
            div_db.append(temp)
        return div_db

    def create_validation_set(self,div_db,i):
        validationSet = div_db[i]
        # print(len(validationSet))
        validationSet = [self.attr_list] + validationSet
        #print(len(validationSet))
        return validationSet


    def fscr_per_turn(self,adaboost_set,validationSet ):
        a = Adaboost(adaboost_set, 5)
        learners, learner_priority = a.algo()
        f = FScore(validationSet, learners, learner_priority)
        fscr,accuracy_per_turn = f.determine_fscore()
        return fscr,accuracy_per_turn

    def validation(self):
        db_ = self.db[1:]
        div_db = self.divided_db(db_)
        #print(len(db_))
        f1Scores = []
        accuracies = []
        for i in range(self.turns):
            print(" turn ",i)
            validationSet = self.create_validation_set(div_db,i)
            #print(len(validationSet))
            adaboost_set = []
            for j in range(self.turns):
                if (j != i):
                    adaboost_set += div_db[j]
            adaboost_set = [self.attr_list] + adaboost_set

            fscr,accuracy_per_turn = self.fscr_per_turn(adaboost_set,validationSet)
            #print(fscr)
            f1Scores.append(fscr)
            accuracies.append(accuracy_per_turn)

        #print(len(f1Scores))
        fscore = sum(f1Scores)/float(self.turns)
        acc_score = sum(accuracies)/float(self.turns)
        print("the f1 score of simulation: ",fscore)
        print("the accuracy of simulation: ", acc_score)












##################################################################################

rd = ReadCSV();
db_1= rd.produceDB()
k = KFoldCross(db_1,5)
#print(k.block_size)
k.validation()
from ReadCSV import  ReadCSV
from Weights import  Weights
import math

class Stump:

    attribute_list = []
    unique_val_attr = []
    list_per_attr = []
    db = []
    main_entropy = 0
    information_gain = []
    dictionary_decisions = {}
    max_info_gain = 0;
    max_info_index = -1;

    def __init__(self,db):
        self.dummy = 0;
        self.db = db

    def define_attr(self):

        numberOfAttr = len(self.db[0])
        list_per_attr = [[] for i in range(numberOfAttr)]

        for i in range( numberOfAttr):
            list_per_attr[i] = [self.db[j][i] for j in range(1,len(self.db))]

        uniqueValueAttr = []
        for i in range(numberOfAttr):
            temp = list_per_attr[i]
            temp = set(temp)
            temp = list(temp)
            #print(temp)
            #list.sort(temp)
            uniqueValueAttr.append(temp)

        attrList = self.db[0]

        self.attribute_list = attrList
        self.list_per_attr = list_per_attr
        self.unique_val_attr = uniqueValueAttr

    def LOG(self,p):
        temp = -p*math.log(p,2)
        return temp
    def determine_main_entropy(self):
        classList = self.list_per_attr[20] #yes no class column values
        #print(classList)
        yes_list = [i for i in classList if i == 'yes']
        no_list = [i for i in classList if i == 'no']
        yes_len = (len(yes_list))
        no_len = (len(no_list))
        sum_ = yes_len+no_len
        yes_prb = yes_len/sum_
        no_prb = no_len/sum_
        #isApple = True if fruit == 'Apple' else False
        #print(yes_prb)
        #print(no_prb)
        yesLog = 0 if yes_prb<=0 else self.LOG(yes_prb)
        noLog = 0 if no_prb<=0 else self.LOG(no_prb)
        sum_ = yesLog + noLog
        self.main_entropy = sum_
        #print(self.main_entropy)

    def information_gain_single(self,index_attr):
        unique_val_of_attr = self.unique_val_attr[index_attr]
        attribute_values = self.list_per_attr[index_attr]
        classList = self.list_per_attr[20]
        #print(unique_val_of_attr)
        #print(attribute_values)

        totalEntropy = 0
        for i in unique_val_of_attr:

            classLen  = len(classList)
            temp_class = []
            for j in range(classLen):

                if(i == attribute_values[j]):
                    temp_class.append(classList[j])

            #print("\n")
            #print("attr_val "+str(i))
            #print(len(temp_class))
            count = len(temp_class)
            yes = temp_class.count('yes')
            no = temp_class.count('no')
            #print(yes)
            #print(no)
            yes_prb = yes / count
            no_prb = no / count
            #print(yes_prb)
            #print(no_prb)
            yesLog = 0 if yes_prb <= 0 else self.LOG(yes_prb)
            noLog = 0 if no_prb <= 0 else self.LOG(no_prb)
            #print(yesLog)
            #print(noLog)
            sum_ = yesLog + noLog
            #print(sum_)
            avgVal = (count/classLen)*sum_
            #print(avgVal)
            totalEntropy += avgVal

        #print("totalus")
        #print(totalEntropy)
        temp_info_gain = self.main_entropy-totalEntropy
        #print(temp_info_gain)
        self.information_gain.append(temp_info_gain)

    def information_gain_all(self):
        self.information_gain = []
        [self.information_gain_single(i) for i in range(len(self.attribute_list)-1)]

    def decisions(self,index_max_info):
        self.dictionary_decisions = {}

        classList = self.list_per_attr[20]
        attribute_values = self.list_per_attr[index_max_info]

        for i in self.unique_val_attr[index_max_info]:

            classLen = len(classList)
            temp_class = []
            for j in range(classLen):

                if (i == attribute_values[j]):
                    temp_class.append(classList[j])

            #print("\n")
            #print("attr_val "+str(i))
            #print(len(temp_class))
            count = len(temp_class)
            yes = temp_class.count('yes')
            no = temp_class.count('no')
            #print(yes)
            #print(no)

            self.dictionary_decisions[i] = 'yes' if yes>=no else 'no'

    def tree(self):
        self.define_attr()
        #print(self.list_per_attr[20])
        self.determine_main_entropy()
        self.information_gain_all()
        #print(self.information_gain)
        #print(len(self.information_gain))

        max_info = max(self.information_gain)
        #print(max_info)
        #print("len of info gain ",len(self.information_gain))
        index_max_info= self.information_gain.index(max_info)
        #print("max index ",self.attribute_list[index_max_info])
        self.decisions(index_max_info)
        self.max_info_gain = max_info
        self.max_info_index = index_max_info

        #print(self.dictionary_decisions)

    def check(self,attr_under_test):
        list_keys = self.dictionary_decisions.keys()
        list_keys = list(list_keys)
        #print(list_keys)
        c = list_keys.count(attr_under_test)
        return  c


    def print_decision(self):
        print(self.dictionary_decisions)

    def decide_for_test(self,row):
        attr_under_test = row[self.max_info_index]
        #print(attr_under_test)
        c = self.check(attr_under_test)
        if c!=0:
            decision = self.dictionary_decisions[attr_under_test]
            return decision
        #print("didn't find")
        return 'no'







####################################MAIN###########################
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

for i in range(10):
    samp = rd.create_sampleList(10, db_1, w_sum)
    s = Stump(samp);
    s.tree();
    s.print_decision()
#print(len(types))
#printList(a)
#print(len(attrName))




'''

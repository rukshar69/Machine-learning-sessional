
import csv
from random import *
from Weights import Weights
class ReadCSV:

    def __init__(self):
        self.dummy = 0;

    def CSVRead(self):
        file = "bank-additional\\bank-additional-full.csv";
        f = open(file, 'r')
        delim = ';'
        reader = csv.reader(f, delimiter=delim)
        rows = []
        for row in reader:
            # print(row)
            rows.append(row)
            length_of_1_row = len(row);
            # print(length_of_1_row)
        '''
        for r in rows:
            print("in rows")
            print(r)
        '''
        f.close()

        # checkType(rows[1])
        return rows

    def modifyDB(self,db):
        header_db = db[0]
        # print(header_db)
        classCol = len(header_db) - 1;
        # print(classCol)
        db_mod = [r for r in db if r[classCol] == 'yes']
        # print(len(db_mod))
        # printList(db_mod)
        db_no = [r for r in db if r[classCol] == 'no']
        # print(len(db_no))
        shuffle(db_no)
        # print(len(db_no))
        db_no = db_no[:len(db_mod)]  # equal to yes
        # print(len(db_no))
        # printList(db_no)
        db_mod = ((db_mod + db_no))
        shuffle(db_mod)
        # print(len(db_mod))
        # printList(db_mod)
        col_to_modify = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
        for r in db_mod:
            for i in col_to_modify:
                r[i] = float(r[i])

        float_list = []

        for i in col_to_modify:
            temp = []
            for r in db_mod:
                temp.append(r[i])
            float_list.append(temp)

        ind = 0
        for r in float_list:
            temp = r;
            min_ = min(temp)
            max_ = max(temp)
            # 4 classes
            bucketSize = (max_ - min_) / 4;
            temp = []
            for i in r:
                a = int((i - min_) / bucketSize);
                temp.append(a)
            float_list[ind] = temp
            ind += 1
            # print(len(temp))
            # print(temp)
            # print(float_list[ind-1])

        len_db_mod = len(db_mod)
        len_col_to_modify = len(col_to_modify)

        for i in range(len_col_to_modify):
            colNum = col_to_modify[i]
            colVect = float_list[i]

            for j in range(len_db_mod):
                db_mod[j][colNum] = colVect[j]

        db_mod = [header_db] + db_mod
        return db_mod

    def produceDB(self):
        db = self.CSVRead()
        db = self.modifyDB(db)
        return db

    def create_sampleList(self,sample_freq,db, sum_w8):

        sample = []
        #sample.append([])
        #sample.append([])


        rand_list = [uniform(0, 1) for x in range(sample_freq)]
        #print(rand_list)
        w = Weights(100)
        for r in rand_list:
            #print(r)
            i = w.index_sum_weight(sum_w8, r)
            #print(i)
            sample.append(db[i])
            #sample[1] = sample[1] + [i]
        header = db[0]

        sample = [header] + sample;
        return sample


#TESTING zzzzzzzzzzzzzzzzzzzzzzzzzzzzz
#rd = ReadCSV();
#db = rd.produceDB()
#print(db[23])
#checkType(rd[23])


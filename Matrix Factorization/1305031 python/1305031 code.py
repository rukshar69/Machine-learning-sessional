import pandas as pd
import numpy as np
import math

train_filename ="ratings_train.xlsx"
test_filename = "ratings_validate.xlsx"


class DataProcess:
    def __init__(self,train_fname, test_fname):
        self.train_fname =train_fname
        self.test_fname =test_fname
        self.train_mat = []
        self.test_mat = []

    def createNewMatrix(self):
        t_data = pd.read_excel(self.train_fname, header=None)
        v_data = pd.read_excel(self.test_fname, header=None)
        self.test_mat = v_data._values
        self.train_mat = t_data._values


class ALS:
    def __init__(self,matrix, user_no, prod_no, latent_factors, lambda_,epoch):
        self.matrix = matrix
        self.user_no = user_no
        self.prod_no = prod_no
        self.latent_factors = latent_factors
        self.factorU = lambda_
        self.epoch = epoch
        self.U = []
        self.VT =[]
        self.convergence = 1e-2
        self.Ik = np.identity(self.latent_factors)
        self.loss_list = []

    def get_x_nm_un(self,m):
        x_m = []
        i = 0
        while i<self.user_no:
            if(self.matrix[i][m]!=-1):
                x_m.append(self.matrix[i][m])
            i+=1

        x_m = np.array([x_m])
        x_m = np.transpose(x_m)
        return  x_m


    def get_u_n(self,m):
        u_n = []
        i = 0
        while i<self.user_no:
            if(self.matrix[i][m]!=-1):
                u_n.append(self.U[i])
            i+=1
        u_n = np.array(u_n)
        u_nTranspose = np.transpose(u_n)
        return u_n , u_nTranspose


    def normal_dist(self,a,b):
        tup = (a,b)
        return np.random.uniform(0,5,tup)
    def initUV(self):
        self.U = self.normal_dist(self.user_no,self.latent_factors)
        self.VT = self.normal_dist(self.latent_factors,self.prod_no)

    def updateV(self):
            result = 0
            i = 0
            while i< self.prod_no:
                u_n, u_nT = self.get_u_n(i)
                x_nm = self.get_x_nm_un(i)
                VmT = np.matmul(np.linalg.inv(np.matmul(u_nT, u_n) + self.factorU * self.Ik), np.matmul(u_nT, x_nm))
                result =  np.concatenate((result, VmT), axis=1) if i!=0 else VmT
                i+=1
            return  result

    def get_v_m(self,n):
        v_m = []
        i = 0
        while i< self.prod_no:
            if(self.matrix[n][i]!= -1):
                v_m.append(self.VT[:,i])
            i+=1
        v_m = np.array(v_m)
        v_mT = np.transpose(v_m)
        return  v_mT, v_m

    def x_nm_for_vm(self,n):
        x_nm = []
        i = 0
        while i< self.prod_no:
            if(self.matrix[n][i]!= -1):
                x_nm.append(self.matrix[n][i])
            i+=1
        x_nm = np.array([x_nm])
        return  x_nm

    def updateU(self):
            result = 0
            i = 0

            while i< self.user_no:
                v_m,v_mT  = self.get_v_m(i)
                x_nm = self.x_nm_for_vm(i)
                Un = np.matmul(np.matmul(x_nm, v_mT), np.linalg.inv(np.matmul(v_m, v_mT) + self.factorU * self.Ik))
                result =  np.concatenate((result, Un)) if i!=0 else Un
                i+=1
            return  result

    def sqr_loss(self):
            i = 0
            sqr = 0
            c = 0
            while i< self.user_no:
                j = 0
                while j< self.prod_no:
                    if (self.matrix[i][j] != -1):
                        a = np.matmul([self.U[i]], np.transpose(np.array(self.VT[:, j])))
                        a = a[0]
                        t = (self.matrix[i][j] - a)
                        sqr += t*t
                        c+=1
                    j+=1

                i+=1
            return sqr,c;

    def regularization_1(self):
        i = 0
        t = []
        while i< self.user_no:
            temp =np.matmul(self.U[i] ,np.transpose(self.U[i]))
            t.append(temp)
            i+=1
        return  sum(t)

    def regularization_2(self):
            t = []
            i = 0
            while i<self.prod_no:
                temp =np.matmul(np.transpose(self.VT[:, i]),self.VT[:, i] )
                t.append(temp)
                i+=1
            return  sum(t)
    def alsAlgorithm(self):
        self.initUV()
        loss_main = 0
        epoch_no = 0
        while epoch_no< self.epoch:
            self.VT = self.updateV()

            self.U = self.updateU()

            sqr ,c  = self.sqr_loss()
            loss = sqr
            reg1, reg2 = self.regularization_1(),self.regularization_2()
            loss += ( self.factorU*reg1 + self.factorU*reg2)
            ERR = math.sqrt(sqr/c)

            print("loss @ epoch ",epoch_no+1,": ",loss)
            self.loss_list.append(loss)

            if abs(loss-loss_main)<self.convergence:
                print("converge")
                break
            epoch_no +=1

        return ERR


class Validate_Test:
    #latent_factor, lamda, valid_db, U_, Vt_
    def __init__(self,val_db, Umat, VTmat , latent_fact, lambda_):
        self.val_db = val_db
        self.latent_fact = latent_fact
        self.Umat = Umat
        self.VTmat  = VTmat
        self.regularizing_fact = lambda_
        self.Ik = np.identity(self.latent_fact)
        self.matrix = []
        self.prod_no= 0

    def get_v_m(self,n):
        v_m = []
        i = 0
        while i< self.prod_no:
            if(self.matrix[n][i]!= -1):
                v_m.append(self.VTmat[:,i])
            i+=1
        v_m = np.array(v_m)
        v_mT = np.transpose(v_m)
        return  v_mT, v_m

    def x_nm_for_vm(self,n):
        x_nm = []
        i = 0
        while i< self.prod_no:
            if(self.matrix[n][i]!= -1):
                x_nm.append(self.matrix[n][i])
            i+=1
        x_nm = np.array([x_nm])
        return  x_nm

    def rms2(self,given, calculated):
        err = []
        c = 0
        i = 0
        given = given[0]
        calculated = calculated[0]
        while i< len(given):
            if given[i] == -1:
                x = 0
            else:
                t = (given[i]-calculated[i])
                err.append(t*t)
                c+=1
            i +=1

        return  sum(err),c

    def validate(self):
        errs  = []
        counts = []

        for i in range(0,len(self.val_db)):
            self.matrix = np.array([self.val_db[i]])
            self.prod_no = self.VTmat.shape[1]
            v_mT,v_m  = self.get_v_m( 0 )
            x_nm = self.x_nm_for_vm(0)
            Un = np.matmul(np.matmul(x_nm, v_m), np.linalg.inv(np.matmul(v_mT, v_m) + self.regularizing_fact * self.Ik))
            error,numberCounted = self.rms2(self.matrix, np.matmul(Un, self.VTmat))
            errs.append(error)
            counts.append(numberCounted)

        return math.sqrt(sum(errs)/sum(counts))

db = DataProcess(train_filename,test_filename)
db.createNewMatrix()
train_db = db.train_mat
train_db = train_db[:100]
valid_db = db.test_mat
valid_db = valid_db[:20]
test_db = db.test_mat[21:40]
latent_factor = 20
lamda = 0.01

print("train ",train_db.shape)
print("test ",valid_db.shape)
user,prod = train_db.shape[0], train_db.shape[1]

epochs =100

algo = ALS(train_db, user, prod,latent_factor, lamda, epochs)
train_err = algo.alsAlgorithm()
U_, Vt_ = algo.U,algo.VT
Val_Tst = Validate_Test(valid_db,U_,Vt_,latent_factor,lamda)
valid_err = Val_Tst.validate()


tst = Validate_Test(test_db,U_,Vt_,latent_factor,lamda)
tst_err = tst.validate()



print("\nerror in training: ",train_err)
print("error in validation ",valid_err)
print("test error ",tst_err)











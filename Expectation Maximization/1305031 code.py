import math
import sklearn.datasets as skd
import numpy as np
import matplotlib.pyplot as plt

class multivariate_gauss_det:
    def __init__(self,sample,mean,covariance):
        self.sample = sample
        self.mean  = mean
        self.covariance = covariance

    def determine_factor_multiVar(self,dimension):
        two_pi = ( np.pi*2 )**dimension
        determinate_cov = np.linalg.det(self.covariance )
        sqrt = np.sqrt( two_pi* determinate_cov)
        inverse = sqrt**-1
        return inverse

    def exp_term(self):
        s =  np.asarray([self.sample]).transpose()
        m = np.asarray([self.mean]).transpose()
        subtract = (s-m)
        transposeTerm =subtract.transpose()

        inverse_cov = np.linalg.inv(self.covariance)
        invCovMultSubtract =  np.matmul(inverse_cov ,subtract)
        finalMult = np.matmul(transposeTerm,invCovMultSubtract)
        exponent = -(1.0/2.0) * finalMult
        return exponent

    def prob_per_sample(self):
        dimension = 2
        factor = self.determine_factor_multiVar(dimension)

        exponent =self.exp_term()
        expoTerm = np.exp(exponent[0][0])
        whole_prob = factor * expoTerm
        return whole_prob

class M_algo:
    def __init__(self,nm_of_dist, sample_per_distribution, features_ship,sampleDB,miu_to_train, cov_to_train,w8_to_train,thr):
        self.nm_of_dist=nm_of_dist
        self.sample_per_distribution=sample_per_distribution
        self.features_ship = features_ship
        self.sampleDB = sampleDB
        self.miu_to_train= miu_to_train
        self.cov_to_train = cov_to_train
        self.w8_to_train = w8_to_train
        self.thr = thr

    def sumOfP(self,Pij):
        s_Pij = []
        i = 0
        while i<self.nm_of_dist:
            j = 0
            temp = 0
            while j<self.sample_per_distribution:
                temp += Pij[i][j]
                j+=1
            s_Pij.append(temp)
            i+=1
        return s_Pij

    def sumOfPX(self,Pij):
        s_Pij = []
        i = 0
        while i<self.nm_of_dist:
            j = 0
            temp = 0
            while j<self.sample_per_distribution:
                mult =  (Pij[i][j] * self.sampleDB [j])
                temp +=mult
                j+=1
            s_Pij.append(temp)
            i+=1
        return s_Pij

    def sum_PXMinusMean(self,Pij):
        s_Pij = []
        i = 0
        while i<self.nm_of_dist:
            j = 0
            sumPX_M = 0
            while j<self.sample_per_distribution:
                s =  np.asarray([self.sampleDB [j]]).transpose()
                m = np.asarray([self.miu_to_train[i]]).transpose()
                subtract = (s-m)
                transposeTerm =subtract.transpose()
                matrix_mult =np.matmul(subtract,transposeTerm)

                sumPX_M += Pij[i][j] *matrix_mult
                j+=1
            s_Pij.append(sumPX_M)
            i+=1
        return s_Pij


    def mstep2(self, Pij):
        newMean = np.zeros(shape=(self.nm_of_dist, self.features_ship))
        newCovariance = np.zeros(shape=(self.nm_of_dist, self.features_ship, self.features_ship))
        newWeight = np.zeros(shape=self.nm_of_dist)

        sum_pij = self.sumOfP(Pij)
        sum_px = self.sumOfPX(Pij)
        sum_px_miu = self.sum_PXMinusMean(Pij)

        i = 0
        while i< self.nm_of_dist:
            newMean[i] = sum_px[i]/sum_pij[i]

            newCovariance[i] = sum_px_miu[i]/sum_pij[i]
            newWeight[i] = sum_pij[i]/self.sample_per_distribution
            i+=1
        return newMean,newCovariance,newWeight




def matrix_print(weight,mean,covariance):
    print()
    print('weight array:')
    print(weight)
    
    print()
    print('mean array:')
    print(mean)

    print()
    print('covariance array:')
    print(covariance)







class plot_graph:
    def __init__(self,samples, dbMiu, dbCov, miu,cov,number_of_dist):
        self.samples = samples
        self.dbMiu = dbMiu
        self.dbCov =dbCov
        self.miu = miu
        self.cov = cov
        self.x_points = []
        self.y_points = []
        self.numberOfDist = number_of_dist

    def plotOneEllipse(self,x_pos,y_pos, a,b,thetas,color):
        size_t = len(thetas)
        x_ = []
        y_ = []
        i = 0;
        while i<size_t:
            x_.append(x_pos+a*math.cos(thetas[i]))
            i+=1

        i = 0;
        while i<size_t:
            y_.append(y_pos+a*math.sin(thetas[i]))
            i+=1


        line_style = '-.'
        plt.plot(x_,y_,linestyle = line_style, color=color)
    def ellipse_draw(self, mean , covariance, color):

        x_pos_a = []
        y_pos_a = []
        a_param = []
        b_param = []

        i = 0;
        while(i<self.numberOfDist):
            x_pos_a.append(mean[i][0])
            i+=1

        i = 0;
        while(i<self.numberOfDist):
            y_pos_a.append(mean[i][1])
            i+=1

        i = 0;
        while(i<self.numberOfDist):
            a_param.append(covariance[i][0][0])
            i+=1

        i = 0;
        while(i<self.numberOfDist):
            b_param.append( covariance[i][1][1])
            i+=1


        i = 0;
        while(i<self.numberOfDist):
            lower = 0
            upper = np.pi * 2
            length = 111
            thetas = [lower + x*(upper-lower)/length for x in range(length)]
            self.plotOneEllipse(x_pos_a[i],y_pos_a[i],a_param[i],b_param[i],thetas,color)
            i+=1

    def populate_xy(self):
        size = len(self.samples)
        self.x_points = [self.samples[i][0] for i in range(size)]
        self.y_points = [self.samples[i][1] for i in range(size)]
    def draw_circle(self):
        self.ellipse_draw(self.dbMiu,self.dbCov,'black')
        self.ellipse_draw(self.miu , self.cov,'red')

    def draw_gr(self):
        self.populate_xy();
        plt.scatter(self.x_points,self.y_points)
        self.draw_circle()
        plt.xlabel("attr 1")
        plt.ylabel("attr 2")
        plt.show()


def take_input():
    #dist_no = int(input("how many distributions: "))
    dist_no = 3
    #sample_per_dist = int(input("how many samples per dist: "))
    sample_per_dist = 400
    return dist_no, sample_per_dist

def handle_params():
    dist_no , sample_per_dist = take_input()
    feature_no = 2;
    low_limit = 1e-7
    return  dist_no,sample_per_dist,feature_no,low_limit
dist_no,sample_freq,number_of_feature,lower_limit_ = handle_params()

class setup_db:
    def __init__(self,num_dist,samp_freq,feature_no):
        self.num_dist = num_dist
        self.samp_freq = samp_freq
        self.feature_no = feature_no

    def init_cov(self):
        #Generate a random symmetric matrix.
        covarianceMatrix = [skd.make_spd_matrix(n_dim=self.feature_no) for i in range(self.num_dist)]
        return covarianceMatrix

    def createDBCov(self):
        covariance = []
        covariance.append(np.array([[0.5,0.3],[0.3,0.5]]))
        covariance.append(np.array([[1.1,0.5],[0.5,1.1]]))
        covariance.append(np.array([[0.3,0.2],[0.2,0.3]]))
        covariance = np.array(covariance)
        return covariance

    def createDBMean(self):
        mean = []
        mean.append(np.array([0,0]))
        mean.append(np.array([0,4]))
        mean.append(np.array([3,3]))
        mean = np.array(mean)
        return  mean


    def create_samples(self,mean,covariance):
        datapoints = []
        i = 0
        while(i<self.samp_freq):
            j = 0
            while(j<self.num_dist):
                sample = np.random.multivariate_normal(mean[j],covariance[j])
                datapoints.append(sample)
                j+=1
            i+= 1

        return  datapoints
    def init_mean_matrix(self):
        meanMatrix = [[np.random.normal() for j in range(self.feature_no)] for i in range(self.num_dist)]
        return meanMatrix

    def init_weights(self):
        prior = 1.0/self.num_dist
        weights = [prior for i in range(self.num_dist)]
        return  weights

    def setup(self):
        dbMean = self.createDBMean()
        dbCov = self.createDBCov()
        samples = self.create_samples(dbMean,dbCov )
        cov= self.init_cov()
        w8 = self.init_weights()
        miu = self.init_mean_matrix()
        return samples,dbMean,dbCov,miu,cov,w8

db = setup_db(dist_no,sample_freq,number_of_feature)
sampleList,datasetMean,datasetCovariance,mean,covariance,weight = db.setup()

class EM_implementation:
    def __init__(self,nm_of_dist, sample_per_distribution, features_ship,sampleDB,miu_to_train, cov_to_train,w8_to_train,thr):
        self.nm_of_dist=nm_of_dist
        self.sample_per_distribution=sample_per_distribution
        self.features_ship = features_ship
        self.sampleDB = sampleDB
        self.miu_to_train= miu_to_train
        self.cov_to_train = cov_to_train
        self.w8_to_train = w8_to_train
        self.thr = thr

    def checkEndCond(self,diff):
         flag = False if diff<self.thr else True
         return flag

    def loglikelihood_distribution(self):

        j = 0
        weighed_n_array = []
        size_db = self.sample_per_distribution
        while(j < size_db):
            weighted_n = []
            i = 0
            while i< self.nm_of_dist:
                multi = multivariate_gauss_det(self.sampleDB[j],self.miu_to_train[i],self.cov_to_train[i])
                weighted_n.append(self.w8_to_train[i] * multi.prob_per_sample())
                i+= 1
            weighed_n_array.append(np.log(sum(weighted_n)))
            j+=1
        log_sum = sum(weighed_n_array)
        return log_sum

    def E(self):
        Pij = []
        i = 0
        while(i< self.nm_of_dist):
            temp = []
            j=0
            while(j<self.sample_per_distribution):
                multi = multivariate_gauss_det(self.sampleDB[j], self.miu_to_train[i],self.cov_to_train [i])
                temp.append(self.w8_to_train[i] * multi.prob_per_sample())
                j+=1
            i += 1
            Pij.append(temp)

        j =0
        while(j<self.sample_per_distribution):
            one_sample_three_dist = []
            i = 0
            while(i< self.nm_of_dist):
                one_sample_three_dist.append(Pij[i][j])
                i+=1

            i = 0
            while(i< self.nm_of_dist):
                Pij[i][j] = Pij[i][j]/sum(one_sample_three_dist)
                i+=1
            j+=1

        return Pij

    def algorithm(self):

        #printMatrices(mean,covariance,weight)

        it = 1
        prevLog = 0
        diff = 101
        while(self.checkEndCond(diff)==True):
            if it%10 == 0:
                print("Iteration: %d"%it)
            Pij = self.E()
            m = M_algo(self.nm_of_dist, self.sample_per_distribution, self.features_ship,self.sampleDB,self.miu_to_train, self.cov_to_train,self.w8_to_train,self.thr)
            self.miu_to_train, self.cov_to_train, self.w8_to_train = m.mstep2( Pij)
            newLog =self.loglikelihood_distribution()
            diff = math.fabs(newLog-prevLog)
            prevLog = newLog
            it+=1
        return self.miu_to_train,self.cov_to_train,self.w8_to_train

matrix_print(weight,mean,covariance,)
em = EM_implementation(dist_no,sample_freq,number_of_feature,sampleList,mean,covariance,weight ,lower_limit_)
mean, covariance,weight = em.algorithm()
matrix_print(weight,mean,covariance,)
gr = plot_graph(sampleList,datasetMean,datasetCovariance,mean,covariance,dist_no)
gr.draw_gr()






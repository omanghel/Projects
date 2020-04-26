import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class processRawData():
    def __init__(self,file):
        self.datafile = file

    def processData(self):
        '''Read the csv file, read the column names, read the number of rows'''
        parameters = pd.read_csv(self.datafile)
        columnsN = parameters.head()
        Nrows = parameters[columnsN.columns[1]].count()

        '''Create a new data frame and keep only the last cycles (primary or sub-cycle)'''
        ### optimization : try parameters.drop() to substitute creating the new dataframe
        newDF = pd.DataFrame(columns = columnsN.columns)
        for i in range(Nrows):
            if i < Nrows-1:
                if parameters.loc[i+1]['Record_Type'] == 'Primary_Cycle':
                    newDF = newDF.append(parameters.loc[i])
                elif parameters.loc[i+1]['Record_Type'] == 'Sub_Cycle':
                    continue
            else:
                newDF = newDF.append(parameters.loc[i])
        #print(newDF)

        '''modify data frame look (row name, column name etc)'''
        newDF.insert(4,'Pregnancy',newDF['initial_outcome'],True)
        newDF['Record_Type'] = newDF['Record_Type'].replace(['Sub_Cycle'],'Primary_Cycle')
        #print(newDF)
        #print(newDF.head().columns)
        return newDF

    def datatoIntMapping(self,data_dframe):
        '''Changing the string features to integers for data preparation for ML algorithms'''
        colNames = data_dframe.head().columns
        for c in colNames:
            if c.find('Feature') != -1:
                data_dframe[c] *= 1
        data_dframe['Pregnancy'] = data_dframe['Pregnancy'].replace(['livebirth'],'1')
        data_dframe['Pregnancy'] = data_dframe['Pregnancy'].replace(['preg_none'],'0')
        return data_dframe

    def plotFeaturesvsPreg(self,data_dframe,f1,f2):
        fig = plt.figure(figsize=(6,6))
        main_ax = fig.add_subplot()
        feat1 = str(f1)
        feat2 = str(f2)
        #separate features vs class
        df_birth = data_dframe[data_dframe['Pregnancy']=='1']
        df_no_preg = data_dframe[data_dframe['Pregnancy']=='0']
        #plot features vs class
        main_ax.plot(df_birth[feat1].astype(float),df_birth[feat2].astype(float),'ok', marker='+',markersize=4,alpha=0.2,color="purple",label='birth')
        main_ax.plot(df_no_preg[feat1].astype(float),df_no_preg[feat2].astype(float),'ok', marker='o',markersize=4,alpha=0.2,color="red",label='no preg')
        main_ax.grid(True)
        main_ax.set_ylabel(feat2)
        main_ax.set_xlabel(feat1)
        main_ax.legend()
        fig.savefig('pregOutcome_for_'+feat1+'vs'+feat2+'.png')

    def plotFeatures(self,data_dframe,f1):
        feat1 = str(f1)
        #separate features vs class
        df_birth = data_dframe[data_dframe['Pregnancy']=='1']
        df_no_preg = data_dframe[data_dframe['Pregnancy']=='0']
        #plot features vs class
        fig,ax = plt.subplots()
        bval,bins,patches = plt.hist(df_birth[feat1].astype(float),bins = 50,facecolor="purple",label='birth')
        bval,bins,patches = plt.hist(df_no_preg[feat1].astype(float),bins = 50,facecolor="red",label='no preg')
        ax.legend()
        plt.xlabel(feat1,fontsize = 16)
        plt.ylabel('Number of primary cycles',fontsize = 16)
        plt.savefig('plot_'+feat1+'.png')


class runAnalysis():
    def __init__(self,trainData,testData,noFeat_=[],_alpha=0.1,_lambda=100.0,_Niter=1000,_epsilon=0.0001, _threshold = 0.6):
        self.trainD = trainData
        self.testD = testData
        self.x = None
        self.y = None
        self.x_test = None
        self.y_true = None
        self.theta_logisticR = None
        self.m = None
        self.n = None
        self.alpha = _alpha
        self.lmda = _lambda
        self.Niter = _Niter
        self.epsilon = _epsilon
        self.threshold = _threshold
        self.noFeat = noFeat_

    def pregRatePerCycle(self,datadf):
        '''Compute pregnancy rates pre cycle'''
        liveB = datadf[datadf['Pregnancy'] == '1']['Pregnancy'].count()
        nopreg = datadf[datadf['Pregnancy'] == '0']['Pregnancy'].count()
        pregnancyRate_perCycle = liveB/(liveB+nopreg)
        #print('Pregnancy rate per cycle = ',pregnancyRate_perCycle)
        return pregnancyRate_perCycle

    def weightedPregRatePerPatient(self,datadf):
        '''Compute pregnancy rates per patient'''
        uniqueID = datadf['ID'].unique()
        countID = []
        liveBid = []
        for i in uniqueID:
            countID.append(datadf[datadf['ID'] == i]['ID'].count())
            liveBid.append(datadf[datadf['ID'] == i][datadf['Pregnancy'] == '1']['ID'].count())
        weight_per_ID = []
        for c in range(len(countID)):
            weight_per_ID.append(liveBid[c]/countID[c]*1.0)
        pregnancyRate_perPatient_weighted = sum(weight_per_ID)/len(weight_per_ID)
        #print('Weighted pregnancy rate per patient = ',pregnancyRate_perPatient_weighted)
        return pregnancyRate_perPatient_weighted

    def xyMatrices(self):
        #preparing the matrix for the training set (used to train the model)
        y = np.array(self.trainD['Pregnancy'])
        self.y = y.astype(np.int)
        self.m = len(self.y)
        features = self.trainD.head().columns[5:]
        excludeFeat =np.array(self.noFeat)
        #excludeFeat = np.array(['Feature_3','Feature_5'])
        features = features[np.logical_not(np.isin(features,excludeFeat))]
        self.n = len(features)
        #print(features)
        x = self.trainD[features].values
        self.x = x.astype(np.float)

        #preparing the matrix for the testing set (used to only test/validate the model)
        yt = np.array(self.testD['Pregnancy'])
        self.y_true = yt.astype(np.int)
        xt = self.testD[features].values
        self.x_test = xt.astype(np.float)

    def customLogisticRegression(self):
        theta = np.ones(self.n)
        self.computeThetas(theta)

        cost = []
        for i in range(self.Niter):
            newtheta,loghSum = self.computeThetas(theta)
            #print(loghSum)
            cost.append((-1/self.m)*loghSum+((self.lmda/(2*self.m))*sum(pow(newtheta,2))))
            #print(cost)
            theta = newtheta
            if i > 0 and abs(cost[i]-cost[i-1]) < self.epsilon:
                break
        self.plotCost(cost)
        self.theta_logisticR = np.array(theta).reshape(-1,1)
        print("Logistic Regression custom - weights: ", self.theta_logisticR)
        hypoth = self.computeSigmoid(self.theta_logisticR,self.x_test)
        #compare this to null error and accuracy from Scikit
        print('Logistic Regression custom - AUC score = ',roc_auc_score(self.y_true,hypoth))
        y_predict = (hypoth > self.threshold).astype(int)
        print('Logistic Regression custom - accuracy_score = ',accuracy_score(self.y_true,y_predict))
        print('Logistic Regression custom - precision_score = ',precision_score(self.y_true,y_predict))
        print('Logistic Regression custom - f1_score = ',f1_score(self.y_true,y_predict))

    def plotCost(self,cost):
        iteration = [i for i in range(len(cost))]
        #fig,ax = plt.subplots()
        fig = plt.figure(figsize=(6,6))
        main_ax = fig.add_subplot()
        main_ax.plot(iteration,cost, 'ok', markersize=2, alpha=0.2, color="indianred")
        #bval,bins,patches = ax.hist(iteration,cost,facecolor='indianred',edgecolor='indianred',label='Cost')
        main_ax.set_xlabel('Number of iterations',fontsize = 16)
        main_ax.set_ylabel('Cost',fontsize = 16)
        fig.savefig('Cost_logisticRegression.png')

    def computeSigmoid(self,theta,x_matrix):
        thetaT = theta.reshape(-1,1)
        h = 1/(1+np.exp(-1*np.matmul(x_matrix,thetaT)))
        return h

    def computeThetas(self,theta):
        h = self.computeSigmoid(theta,self.x)
        hminy = 0
        hminyx = np.zeros(self.n)
        loghSum = 0

        for i in range(1,self.m):
            hminy += h[i][0]-self.y[i]
            loghSum += self.y[i]*np.log(h[i][0]) + (1-self.y[i])*np.log(1-h[i][0])
            for j in range(1,self.n):
                hminyx[j] += (h[i][0]-self.y[i])*self.x[i][j]

        theta[0] = theta[0] - self.alpha*(1/self.m)*hminy
        for j in range(1,self.n):
            theta[j] = theta[j]*(1-(self.alpha*self.lmda/self.m)) - self.alpha*(1/self.m)*hminyx[j]

        newth = np.array(theta)
        #print('---------loghSum = ',loghSum)
        return newth,loghSum

    def predictSVM_WithScikit(self):
        model = SVC(kernel='rbf',gamma = 'scale',C=1.0).fit(self.x,self.y)
        print('SVC: Scikit model - score = ',model.score(self.x,self.y))
        #compare the prediction to the true
        print('SVC: AUC score = ',roc_auc_score(self.y_true, model.predict(self.x_test)))
        print('SVC: accuracy_score = ',accuracy_score(self.y_true,model.predict(self.x_test)))
        print('SVC: precision_score = ',precision_score(self.y_true,model.predict(self.x_test)))
        print('SVC: f1_score = ',f1_score(self.y_true,model.predict(self.x_test)))

    def predictLG_WithScikit(self):
        model = LogisticRegression().fit(self.x,self.y)
        print('LogR: Scikit model - score = ',model.score(self.x,self.y))
        print('LogR: Scikit null error (mean) = ',self.y.mean())
        print('LogR: Scikit model - thetas = [',model.intercept_,', ',model.coef_,']')
        #compare the prediction to the true
        print('LogR: AUC score = ',roc_auc_score(self.y_true, model.predict(self.x_test)))
        print('LogR: accuracy_score = ',accuracy_score(self.y_true,model.predict(self.x_test)))
        print('LogR: precision_score = ',precision_score(self.y_true,model.predict(self.x_test)))
        print('LogR: f1_score = ',f1_score(self.y_true,model.predict(self.x_test)))


if __name__ == '__main__':
    dataTrain = '/Users/ioana/Desktop/Documents/univfy/train_final.csv'
    dataTest = '/Users/ioana/Desktop/Documents/univfy/test_final.csv'

    '''Prepare data for analysis'''
    train = processRawData(dataTrain)
    dataProcessed = train.processData()
    print(dataProcessed)
    dataTrainML = train.datatoIntMapping(dataProcessed)
    print(dataTrainML)

    test = processRawData(dataTest)
    dataTestProcessed = test.processData()
    print(dataTestProcessed)
    dataTestML = test.datatoIntMapping(dataTestProcessed)
    print(dataTestML)

    '''Plotting some features for training data'''
    featList = dataTrainML.head().columns[5:]
    print(featList)
    train.plotFeaturesvsPreg(dataTrainML,'Feature_3','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_16','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_15','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_1','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_9','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_10','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_11','Feature_5')
    train.plotFeaturesvsPreg(dataTrainML,'Feature_12','Feature_5')

    for i in range(len(featList)):
        train.plotFeatures(dataTrainML,featList[i])
        train.plotFeaturesvsPreg(dataTrainML,featList[i],'Pregnancy')
        #train.plotFeaturesvsPreg(dataTrainML,featList[i],featList[i+1])
        plt.close("all")

    '''Run the analysis (train algorithm and predict results)'''
    #ana = runAnalysis(dataTrainML,dataTestML)
    ana = runAnalysis(dataTrainML,dataTestML,['Feature_3','Feature_5'])
    #ana = runAnalysis(dataTrainML,dataTestML,['Feature_3'])
    print('Training Data - Pregnancy rate per cycle = ',ana.pregRatePerCycle(dataTrainML))
    print('Testing Data - Pregnancy rate per cycle = ',ana.pregRatePerCycle(dataTestML))
    print('Training Data - Weighted pregnancy rate per patient = ',ana.weightedPregRatePerPatient(dataTrainML))
    print('Testing Data - Weighted pregnancy rate per patient = ',ana.weightedPregRatePerPatient(dataTestML))
    ana.xyMatrices()
    print('*****Custom Logistic Regression*****')
    ana.customLogisticRegression()
    print('*****SVM Classifier*****')
    ana.predictSVM_WithScikit()
    print('*****Logistic Regression Classifier*****')
    ana.predictLG_WithScikit()

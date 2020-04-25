import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class pseudoE():
    def __init__(self,fileP,threshold_= 0.8):
        self.file = fileP
        self.data = None
        self.y = None
        self.x = None
        self.x_test = None
        self.y_true = None
        self.trainD = None
        self.testD = None
        self.threshold = threshold_

    def readFile(self):
        self.data = pd.read_csv(self.file)

    def makeSamples(self,df,thrshd):
        trainS = df.sample(frac = thrshd)
        testS = df.drop(trainS.index)
        return trainS,testS

    def downSampling(self,rsvalue,dataDF):
        c1 = dataDF[dataDF['Treatment_sought'] == 1]
        c2 = dataDF[dataDF['Treatment_sought'] == 0]
        print('before sampling: ', c1.shape,', ',c2.shape)
        m1,n1 = c1.shape
        m2,n2 = c2.shape
        #print(m2/m1)
        c1 = c1.sample(random_state = rsvalue,frac = m2/m1)
        print('after sampling: ', c1.shape,', ',c2.shape)

        df_balanced = c1.append(c2)
        #print(df_balanced.head())
        #print(df_balanced.shape)
        return df_balanced

    def runPseudoExp(self,Nexp):
        columnsMetrics = ['auc','f1']
        rf_df = pd.DataFrame(columns=columnsMetrics)
        log_df = pd.DataFrame(columns=columnsMetrics)
        svm_df = pd.DataFrame(columns=columnsMetrics)
        for iexp in range(Nexp):
            trainDimbalanced,testD = self.makeSamples(self.data,self.threshold)
            trainD = self.downSampling(iexp,trainDimbalanced)
            print('train sample = ',trainD.shape,', test sample = ',testD.shape)
            x_train,y_train,x_test,y_test = self.xyMatrices(trainD,testD)
            rf_auc,rf_f1 = self.predictRF_WithScikit(x_train,y_train,x_test,y_test)
            log_auc,log_f1 = self.predictLG_WithScikit(x_train,y_train,x_test,y_test)
            svm_auc,svm_f1 = self.predictSVM_WithScikit(x_train,y_train,x_test,y_test)
            rf_df_i = pd.DataFrame({'auc':[rf_auc],'f1':[rf_f1]},columns = columnsMetrics)
            log_df_i = pd.DataFrame({'auc':[log_auc],'f1':[log_f1]},columns = columnsMetrics)
            svm_df_i = pd.DataFrame({'auc':[svm_auc],'f1':[svm_f1]},columns = columnsMetrics)
            rf_df = rf_df.append(rf_df_i,ignore_index = True)
            log_df = log_df.append(log_df_i,ignore_index = True)
            svm_df = svm_df.append(svm_df_i,ignore_index = True)
        return rf_df,log_df,svm_df

    def xyMatrices(self,trainD,testD):
        ytr = np.array(trainD['Treatment_sought'])
        y_train = ytr.astype(np.int)
        dataTrain = trainD.drop(columns = ['Unnamed: 0','Treatment_sought','Is_tech','Have_MHD','Work_interference_with_Treatment','Work_interference_without_Treatment'])
        dataUsed = dataTrain.drop(columns = ['Gender'])
        #dataUsed = dataTrain.drop(columns = ['Gender','Past_MHD'])
        print(dataUsed.head())
        #print(dataUsed.shape)

        features = dataUsed.columns
        #print('training: ',dataUsed[features].shape)
        xtr = dataUsed[features].values
        x_train = xtr.astype(np.float)

        #preparing the matrix for the testing set (used to only test/validate the model)
        dataTesting = testD
        yt = np.array(dataTesting['Treatment_sought'])
        y_test = yt.astype(np.int)
        #print('testing: ',dataTesting[features].shape)
        xt = dataTesting[features].values
        x_test = xt.astype(np.float)

        return x_train,y_train,x_test,y_test

    def predictSVM_WithScikit(self,x,y,x_test,y_true):
        svcmodel = SVC(kernel='rbf',gamma = 'scale',C=1.0).fit(x,y)
        svc_ypred = svcmodel.predict(x_test)
        svc_f1 = f1_score(y_true, svc_ypred)
        svc_auc = roc_auc_score(y_true, svc_ypred)
        #svc_yprob = svcmodel.predict_proba(x_test)
        #svc_yprob = [p[1] for p in svc_yprob]
        #svc_auc = roc_auc_score(y_true,svc_yprob)
        return svc_auc,svc_f1

    def predictLG_WithScikit(self,x,y,x_test,y_true):
        logmodel = LogisticRegression(max_iter=1000).fit(x,y)
        log_ypred = logmodel.predict(x_test)
        log_f1 = f1_score(y_true, log_ypred)
        #log_auc = roc_auc_score(y_true, log_ypred)
        log_yprob = logmodel.predict_proba(x_test)
        log_yprob = [p[1] for p in log_yprob]
        log_auc = roc_auc_score(y_true,log_yprob)
        return log_auc,log_f1

    def predictRF_WithScikit(self,x,y,x_test,y_true):
        rfmodel = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt').fit(x,y)
        rf_ypred = rfmodel.predict(x_test)
        rf_f1 = f1_score(y_true, rf_ypred)
        #rf_auc = roc_auc_score(y_true, rf_ypred)
        rf_yprob = rfmodel.predict_proba(x_test)
        rf_yprob = [p[1] for p in rf_yprob]
        rf_auc = roc_auc_score(y_true,rf_yprob)
        return rf_auc,rf_f1

    def makePlots(self,rf_df,log_df,svm_df):
        fig,ax = plt.subplots()
        print('RF: AUC-mean = ',rf_df['auc'].mean())
        print('Log: AUC-mean = ',log_df['auc'].mean())
        print('SVM: AUC-mean = ',svm_df['auc'].mean())
        colors = ["darkorange","darkgreen","grey"]

        #sns.set_style('darkgrid')
        sns.set_color_codes()
        sns.distplot(rf_df['auc'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[0], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[0]}, label = 'Random Forest: '+str(round(rf_df['auc'].astype(float).mean(),2)))
        sns.distplot(log_df['auc'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[1], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[1]}, label = 'Logistic Reg: '+str(round(log_df['auc'].astype(float).mean(),2)))
        sns.distplot(svm_df['auc'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[2], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[2]}, label = 'SVM: '+str(round(svm_df['auc'].astype(float).mean(),2)))

        ax.legend()
        plt.xlabel('AUC',fontsize = 16)
        plt.ylabel('Entries',fontsize = 16)
        textL = [rf_df['auc'].mean(),log_df['auc'].mean(),svm_df['auc'].mean()]
        #textL = ['aa','bb','cc']
        i = 0
        for il in range(len(textL)):
            plt.text(-1, 400-i*20, textL[il], color = colors[il])
            i += 1
        plt.savefig('plots_qA/plot_pseudoExp_AUC.png')

        fig,ax = plt.subplots()
        print('RF: F1-mean = ',rf_df['f1'].mean())
        print('Log: F1-mean = ',log_df['f1'].mean())
        print('SVM: F1-mean = ',svm_df['f1'].mean())
        #sns.set_style('darkgrid')
        sns.set_color_codes()
        sns.distplot(rf_df['f1'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[0], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[0]}, label = 'Random Forest: '+str(round(rf_df['f1'].astype(float).mean(),2)))
        sns.distplot(log_df['f1'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[1], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[1]}, label = 'Logistic Reg: '+str(round(log_df['f1'].astype(float).mean(),2)))
        sns.distplot(svm_df['f1'].astype(float), fit=stats.norm, kde = False,
                     fit_kws={"color": colors[2], "lw": 3},
                     hist_kws={"linewidth": 3, "color":colors[2]}, label = 'SVM: '+str(round(svm_df['f1'].astype(float).mean(),2)))
        ax.legend()
        plt.xlabel('F1_score',fontsize = 16)
        plt.ylabel('Entries',fontsize = 16)
        i = 0
        textL = []
        for il in textL:
            plt.text(-1.0, 400-i*20, il)
            i += 1
        plt.savefig('plots_qA/plot_pseudoExp_F1.png')

if __name__ == '__main__':
    dataqAFile = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqA.csv'

    ana = pseudoE(dataqAFile)
    '''
    ana.readFile()
    rf_dataFrame,log_dataFrame,svm_dataFrame = ana.runPseudoExp(1000)

    print('***** Random Forest *****')
    print(rf_dataFrame.shape)
    print(rf_dataFrame.head())
    print('***** Logistic Regression *****')
    print(log_dataFrame.shape)
    print(log_dataFrame.head())
    print('***** SVM *****')
    print(svm_dataFrame.shape)
    print(svm_dataFrame.head())
    rf_dataFrame.to_csv('rf_qA_results.csv')
    log_dataFrame.to_csv('log_qA_results.csv')
    svm_dataFrame.to_csv('svm_qA_results.csv')
    '''
    rf_dataFrame = pd.read_csv('rf_qA_results.csv')
    log_dataFrame = pd.read_csv('log_qA_results.csv')
    svm_dataFrame = pd.read_csv('svm_qA_results.csv')
    ana.makePlots(rf_dataFrame,log_dataFrame,svm_dataFrame)

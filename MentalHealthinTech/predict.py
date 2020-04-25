import numpy as np
import matplotlib.pyplot as plt
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

class prediction():
    def __init__(self,trainF,testF,noFeat_=[]):
        self.trainfile = trainF
        self.testfile = testF
        self.data = None
        self.testD = None
        self.y = None
        self.x = None
        self.m = None
        self.n = None
        self.x_test = None
        self.y_true = None
        self.noFeat = noFeat_

    def readFile(self):
        self.data = pd.read_csv(self.trainfile)
        self.testD = pd.read_csv(self.testfile)

    def makePlot(self,f1,f2,textL,labelx):
        feat1 = str(f1)
        feat2 = str(f2)

        fig,ax = plt.subplots()
        bval,bins,patches = plt.hist(self.data[feat1].astype(float),bins = 50,facecolor="purple",label = feat1)
        bval,bins,patches = plt.hist(self.data[feat2].astype(float),bins = 50,facecolor="red",label = feat2)
        ax.legend()
        plt.xlabel(labelx,fontsize = 16)
        plt.ylabel('Number of employees',fontsize = 16)
        i = 0
        for il in textL:
            plt.text(-1.0, 400-i*20, il)
            i += 1
        plt.savefig('plots_qA/plot_'+feat1+'_and_'+feat2+'.png')


    def plotFeaturesqA(self,f1):
        #print(self.data.head())
        feat1 = str(f1)
        print(feat1)
        print(self.data[feat1].unique())
        #separate features vs class
        df_soughtT = self.data[self.data['Treatment_sought'] == 1]
        df_not_soughtT = self.data[self.data['Treatment_sought'] == 0]
        print('Treatment_sought = ',df_soughtT.shape)
        print('Treatment_not_sought = ',df_not_soughtT.shape)
        #plot features vs class
        fig,ax = plt.subplots()
        bval,bins,patches = plt.hist(df_soughtT[feat1].astype(float),bins = 50,facecolor="purple",label='sought Treatm')
        bval,bins,patches = plt.hist(df_not_soughtT[feat1].astype(float),bins = 50,facecolor="red",label='did not sought Treatm')
        ax.legend()
        plt.xlabel(feat1,fontsize = 16)
        plt.ylabel('Number of employees',fontsize = 16)
        plt.savefig('plots_qA/plot_'+feat1+'_smote.png')

    def xyMatrices(self):
        #preparing the matrix for the training set (used to train the model)
        dataTraining = self.data
        #dataTraining = self.downSampling(self.data)
        y = np.array(dataTraining['Treatment_sought'])
        self.y = y.astype(np.int)
        self.m = len(self.y)

        dataUsed = dataTraining.drop(columns = ['Unnamed: 0','Treatment_sought','Is_tech','Have_MHD','Work_interference_with_Treatment','Work_interference_without_Treatment'])
        #dataUsed = dataUsed[dataUsed[''] == -1]
        #dataUsed = dataUsed[dataUsed['Gender'] == 0]
        dataUsed = dataUsed.drop(columns = ['Gender'])
        #dataUsed = dataUsed.drop(columns = ['Gender','Past_MHD'])
        print(dataUsed.head())
        print(dataUsed.shape)

        features = dataUsed.columns
        for ifeat in features:
            print(ifeat)
        excludeFeat =np.array(self.noFeat)
        #excludeFeat = np.array(['Feature_3','Feature_5'])
        features = features[np.logical_not(np.isin(features,excludeFeat))]
        #print(features)
        self.n = len(features)
        #print(features)
        print('training: ',dataUsed[features].shape)
        x = dataUsed[features].values
        self.x = x.astype(np.float)
        print(self.x)

        #self.x,self.y = self.upsampling_ROS(self.x,self.y)
        #self.x,self.y = self.upsampling_SMOTE(self.x,self.y)

        #preparing the matrix for the testing set (used to only test/validate the model)
        dataTesting = self.testD
        #dataTesting = self.downSampling(self.testD)
        yt = np.array(dataTesting['Treatment_sought'])
        self.y_true = yt.astype(np.int)
        print('testing: ',dataTesting[features].shape)
        xt = dataTesting[features].values
        self.x_test = xt.astype(np.float)

    def downSampling(self,dataDF):
        c1 = dataDF[dataDF['Treatment_sought'] == 1]
        c2 = dataDF[dataDF['Treatment_sought'] == 0]
        print(c1.shape)
        print(c2.shape)
        m1,n1 = c1.shape
        m2,n2 = c2.shape
        print(m2/m1)
        c1 = c1.sample(frac = m2/m1)
        print(c1.shape)

        df_balanced = c1.append(c2)
        print(df_balanced.head())
        print(df_balanced.shape)
        return df_balanced

    def upsampling_ROS(self,X,y):
        print('Original dataset shape %s' % Counter(y))
        ros = RandomOverSampler(random_state=0)
        X_res, y_res = ros.fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y_res))
        print(X_res)
        print(y_res)
        return X_res,y_res

    def upsampling_SMOTE(self,X,y):
        print('Original dataset shape %s' % Counter(y))
        #X_res, y_res = SMOTE().fit_resample(X, y)
        X_res, y_res = SMOTENC(random_state=0, categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y_res))
        print(X_res)
        print(y_res)
        return X_res,y_res

    def predictSVM_WithScikit(self):
        model = SVC(kernel='rbf',gamma = 'scale',C=1.0).fit(self.x,self.y)
        #model = SVC().fit(self.x,self.y)
        print('SVC: Scikit model - score = ',model.score(self.x,self.y))
        #compare the prediction to the true
        print('SVC: AUC score = ',roc_auc_score(self.y_true, model.predict(self.x_test)))
        print('SVC: accuracy_score = ',accuracy_score(self.y_true,model.predict(self.x_test)))
        print('SVC: precision_score = ',precision_score(self.y_true,model.predict(self.x_test)))
        print('SVC: f1_score = ',f1_score(self.y_true,model.predict(self.x_test)))

    def predictLG_WithScikit(self):
        model = LogisticRegression(max_iter=1000).fit(self.x,self.y)
        print('LogR: Scikit model - score = ',model.score(self.x,self.y))
        print('LogR: Scikit null error (mean) = ',self.y.mean())
        print('LogR: Scikit model - thetas = [',model.intercept_,', ',model.coef_,']')
        #compare the prediction to the true
        print('LogR: AUC score = ',roc_auc_score(self.y_true, model.predict(self.x_test)))
        print('LogR: accuracy_score = ',accuracy_score(self.y_true,model.predict(self.x_test)))
        print('LogR: precision_score = ',precision_score(self.y_true,model.predict(self.x_test)))
        print('LogR: f1_score = ',f1_score(self.y_true,model.predict(self.x_test)))

    def predictRF_WithScikit(self):
        model = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt').fit(self.x,self.y)
        print('RF: Scikit model - score = ',model.score(self.x,self.y))
        #print('RF: Scikit null error (mean) = ',self.y.mean())
        #print('RF: Scikit model - thetas = [',model.intercept_,', ',model.coef_,']')
        #compare the prediction to the true
        print('RF: AUC score = ',roc_auc_score(self.y_true, model.predict(self.x_test)))
        print('RF: accuracy_score = ',accuracy_score(self.y_true,model.predict(self.x_test)))
        print('RF: precision_score = ',precision_score(self.y_true,model.predict(self.x_test)))
        print('RF: f1_score = ',f1_score(self.y_true,model.predict(self.x_test)))

if __name__ == '__main__':
    #train_dataAll = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataAll.csv'
    #train_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqA.csv'
    #train_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqA_balanced_ds.csv'
    #train_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqA_balanced_ros.csv'
    #train_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqA_balanced_smote.csv'
    train_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/test/train_dataqA_balanced_smote.csv'

    #train_dataqB = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/train_dataqB.csv'

    #test_dataAll = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/test_dataAll.csv'
    test_dataqA = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/test/test_dataqA.csv'
    #test_dataqB = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/test_dataqB.csv'

    features = ['Company_size',
                'Is_tech',
                'MentalH_insurance',
                'MentalH_company_info',
                'MentalH_company_resources',
                'Anonymity_protection',
                'Medical_leave_MH',
                'Superviser_comfortable_talking',
                'Coworkers_comfortable_talking',
                'Have_MHD',
                'Past_MHD',
                'Treatment_sought',
                'Family_history_MHD',
                'Work_interference_with_Treatment',
                'Work_interference_without_Treatment',
                'Discouranged_in_revealing_MHD_at_work',
                'MHD_identification_effect_on_career',
                'MHD_identification_team_reaction',
                'Age',
                'Gender']

    #all data
    ana = prediction(train_dataqA,test_dataqA)
    ana.readFile()
    #ana.fixUnbalanced(ana.data)
    #ana.fixUnbalanced(ana.testD)


    #make plots for visualization
    listLabel = ['1 - Often','2 - Sometimes','3 - Rarely','4 - Never','-1 - N/A']
    ana.makePlot('Work_interference_with_Treatment','Work_interference_without_Treatment',listLabel,'Work Interference')

    for ifeat in features:
        if ifeat != 'Treatment_sought':
            ana.plotFeaturesqA(ifeat)
        plt.close("all")


    #Run the analysis (train algorithm and predict results)
    ana.xyMatrices()
    print('*****SVM Classifier*****')
    ana.predictSVM_WithScikit()
    print('*****Logistic Regression Classifier*****')
    ana.predictLG_WithScikit()
    print('*****Random Forest Classifier*****')
    ana.predictRF_WithScikit()

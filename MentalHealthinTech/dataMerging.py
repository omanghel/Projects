import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

class mergeData():
    def __init__(self,fileL):
        self.fileList = fileL
        self.data= None

    def mergeFiles(self):

        keepCol = ['Company_size',
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

        '''Read the csv files'''
        df_all = pd.DataFrame(columns = keepCol)
        for ifile in self.fileList:
            df = pd.read_csv(ifile)
            columnsN = df.columns

            colDrop = [ic for ic in columnsN if not(ic in keepCol)]
            print(colDrop)
            df = df.drop(columns = colDrop)
            print('--------',ifile)
            for icol in df.head():
                print(icol)
            print(df.shape)

            df_all = df_all.append(df, ignore_index = True)

        self.data = df_all

        return self.data

    def makeSamples(self,df,threshold):
        trainS = df.sample(frac = threshold)
        testS = df.drop(trainS.index)

        return trainS,testS

    def downsampling(self,dataDF):
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

    def upsampling_ROS(self,dataDF):
        y = np.array(dataDF['Treatment_sought'])
        y = y.astype(np.int)
        indexTs = dataDF.columns.get_loc('Treatment_sought')
        dataX = dataDF.drop(columns = ['Treatment_sought'])
        cols = dataX.columns
        X = dataX.values
        X = X.astype(np.float)

        print('Original dataset shape %s' % Counter(y))
        ros = RandomOverSampler(random_state=0)
        X_res, y_res = ros.fit_resample(X, y)
        print('ROS - Resampled dataset shape %s' % Counter(y_res))
        print(X_res)
        print(y_res)

        dict_all = {cols[i]:X[:,i] for i in range(len(cols))}
        dataDF_res = pd.DataFrame(X_res,columns= cols)
        dataDF_res.insert(indexTs,'Treatment_sought',y_res,True)

        return dataDF_res

    def upsampling_SMOTE(self,dataDF):
        print('inside smote.... ')
        print(dataDF.tail())
        y = np.array(dataDF['Treatment_sought'])
        y = y.astype(np.int)
        indexTs = dataDF.columns.get_loc('Treatment_sought')
        dataX = dataDF.drop(columns = ['Treatment_sought'])
        cols = dataX.columns
        print('inside smote.... ',cols)
        X = dataX.values
        X = X.astype(np.float)
        indxList = []
        for ic in cols:
            indx = dataX.columns.get_loc(ic)
            indxList.append(indx)
            print(ic)
        print(indxList)

        print('Original dataset shape %s' % Counter(y))
        #X_res, y_res = SMOTENC(sampling_strategy='minority', random_state=42).fit_resample(X, y)
        X_res, y_res = SMOTENC(random_state=0, categorical_features=[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18]).fit_resample(X, y)
        print('SMOTE - Resampled dataset shape %s' % Counter(y_res))
        print(X_res)
        print(y_res)

        dict_all = {cols[i]:X[:,i] for i in range(len(cols))}
        dataDF_res = pd.DataFrame(X_res,columns= cols)
        dataDF_res.insert(indexTs,'Treatment_sought',y_res,True)
        print(dataDF_res.tail())
        return dataDF_res


if __name__ == '__main__':
    dataL_all = ['/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataAll_2019.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataAll_2018.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataAll_2017.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataAll_2016.csv']

    dataL_qA = ['/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqA_2019.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqA_2018.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqA_2017.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqA_2016.csv']

    dataL_qB = ['/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqB_2019.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqB_2018.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqB_2017.csv',
             '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/codes/data/dataqB_2016.csv']

    #dataTrain = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/OSMI_2018.csv'
    dataTest = '...'

    '''Prepare data for analysis'''
    # set threshold for train test: 80/20
    th = 0.8
    #all data
    process_all = mergeData(dataL_all)
    dataAll = process_all.mergeFiles()
    print('data_all = ',dataAll.shape)
    dataAll = dataAll.replace({'Gender':{'Non-binary and gender fluid':'2'}})
    train_dataAll, test_dataAll = process_all.makeSamples(dataAll,th)
    print('train_dataAll = ',train_dataAll.shape)
    print('test_dataAll = ',test_dataAll.shape)
    ds_balancedTraining = process_all.downsampling(train_dataAll)
    ros_balancedTraining = process_all.upsampling_ROS(train_dataAll)
    smote_balancedTraining = process_all.upsampling_SMOTE(train_dataAll)
    ds_balancedTraining.to_csv('train_dataAll_balanced_ds.csv')
    ros_balancedTraining.to_csv('train_dataAll_balanced_ros.csv')
    smote_balancedTraining.to_csv('train_dataAll_balanced_smote.csv')
    train_dataAll.to_csv('train_dataAll.csv')
    test_dataAll.to_csv('test_dataAll.csv')
    dataAll.to_csv('dataAll.csv')


    # data for question A
    process_qA = mergeData(dataL_qA)
    dataqA = process_qA.mergeFiles()
    print('data_qA = ',dataqA.shape)
    dataqA = dataqA.replace({'Gender':{'Non-binary and gender fluid':'2'}})
    train_dataqA, test_dataqA = process_qA.makeSamples(dataqA,th)
    print('train_dataqA = ',train_dataqA.shape)
    print('test_dataqA = ',test_dataqA.shape)
    ds_balancedTraining_qA = process_qA.downsampling(train_dataqA)
    ros_balancedTraining_qA = process_qA.upsampling_ROS(train_dataqA)
    smote_balancedTraining_qA = process_qA.upsampling_SMOTE(train_dataqA)
    ds_balancedTraining_qA.to_csv('train_dataqA_balanced_ds.csv')
    ros_balancedTraining_qA.to_csv('train_dataqA_balanced_ros.csv')
    smote_balancedTraining_qA.to_csv('train_dataqA_balanced_smote.csv')
    train_dataqA.to_csv('train_dataqA.csv')
    test_dataqA.to_csv('test_dataqA.csv')
    dataqA.to_csv('dataqA.csv')

    # data for question B
    process_qB = mergeData(dataL_qB)
    dataqB = process_qB.mergeFiles()
    print('data_qB = ',dataqB.shape)
    dataqB = dataqB.replace({'Gender':{'Non-binary and gender fluid':'2'}})
    train_dataqB, test_dataqB = process_qB.makeSamples(dataqB,th)
    print('train_dataqB = ',train_dataqB.shape)
    print('test_dataqB = ',test_dataqB.shape)
    #balancedTraining_qB = process_qB.downsampling(train_dataqB)
    print(dataqB.head())
    print(train_dataqB.head())
    print(test_dataqB.head())
    #print(balancedTraining_qB.head())
    #balancedTraining_qB.to_csv('train_dataqB_balanced.csv')
    train_dataqB.to_csv('train_dataqB.csv')
    test_dataqB.to_csv('test_dataqB.csv')
    dataqB.to_csv('dataqB.csv')

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
        self.rawData = None

    def fixNan(self,colList,df,value):
        for icol in colList:
            if (len(df[icol].isna()) > 0):
                df[icol].fillna(value, inplace=True)
                #df = df.replace({colName:{'nan':value}})
        return df

    def processData(self):
        '''Read the csv file, read the column names, read the number of rows'''
        parameters = pd.read_csv(self.datafile)
        columnsN = parameters.columns
        print(parameters.head())

        for ic in columnsN:
            print(ic)
            #if parameters[ic].dtypes == bool:
            #    print(ic, ' - ', parameters[ic].dtypes)
        print(parameters.shape)

        df_0 = parameters[parameters['What country do you work in?'] == 'United States of America']
        #mapping bools to 0/1
        df_0 *= 1
        print('in US: ', df_0.shape)
        print(df_0.head())

        df_1 = df_0[df_0['Are you self-employed?'] == 0]
        print('in US, company: ', df_1.shape)
        df_us = df_1[df_1['Is your employer primarily a tech company/organization?'] == 1]
        #df_us = df_2[df_2['Is your primary role within your company related to tech/IT?'] == '1']
        print('in US, company, tech role: ', df_us.shape)

        keepCol = ['How many employees does your company or organization have?',
                  'Is your employer primarily a tech company/organization?',
                  'Does your employer provide mental health benefits as part of healthcare coverage?',
                  'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
                  'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
                  'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
                  'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:',
                  'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
                  'Would you feel comfortable discussing a mental health disorder with your coworkers?',
                  'Do you currently have a mental health disorder?',
                  'Have you had a mental health disorder in the past?',
                  'Have you ever sought treatment for a mental health issue from a mental health professional?',
                  'Do you have a family history of mental illness?',
                  'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?',
                  'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?',
                  'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?',
                  'Do you feel that being identified as a person with a mental health issue would hurt your career?',
                  'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?',
                  'Do you work remotely?',
                  'What is your age?',
                  'What is your gender?']

        colNames = ['Company_size',
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
                    'Work_remotely',
                    'Age',
                    'Gender']

        df_us = df_us.rename(columns=dict(zip(keepCol,colNames)))
        print(df_us.head())
        print(df_us.shape)

        colDrop = [i for i in columnsN if not(i in keepCol)]
        print(colDrop)
        df_us = df_us.drop(columns = colDrop)
        print(df_us.head())

        icol = 'Work_interference_with_Treatment'
        print(df_us[icol].shape)
        print((df_us[icol].dropna()).shape)
        print(df_us[icol].unique())
        print(df_us[df_us[icol].isin(['Yes','Possibly'])].shape)

        #print(df_us[icol])

        #fixing categorical variables
        df_us = df_us.replace({'Company_size':{'1-5':'1','6-25':'2','26-100':'3','100-500':'4','500-1000':'5','More than 1000':'6'}})
        df_us = df_us.replace({'MentalH_insurance':{'Yes':'1','I don\'t know':'2','No':'0','Not eligible for coverage / N/A':'3'}})
        df_us = df_us.replace({'MentalH_company_info':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'MentalH_company_resources':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Anonymity_protection':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Medical_leave_MH':{'Difficult':'0','Very difficult':'0','Somewhat difficult':'1','Neither easy nor difficult':'2','Somewhat easy':'3','Very easy':'4','I don\'t know':'-1'}})
        df_us = df_us.replace({'Superviser_comfortable_talking':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Coworkers_comfortable_talking':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Have_MHD':{'Yes':'1','Possibly':'1','Maybe':'1','Don\'t Know':'2','No':'0'}})
        df_us = df_us.replace({'Past_MHD':{'Yes':'1','Possibly':'1','Maybe':'1','Don\'t Know':'2','No':'0'}})
        df_us = df_us.replace({'Family_history_MHD':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Work_interference_with_Treatment':{'Often':'1','Sometimes':'2','Rarely':'3','Never':'4','Not applicable to me':'-1'}})
        df_us = df_us.replace({'Work_interference_without_Treatment':{'Often':'1','Sometimes':'2','Rarely':'3','Never':'4','Not applicable to me':'-1'}})
        df_us = df_us.replace({'Discouranged_in_revealing_MHD_at_work':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Gender':{'Male':'0','male':'0','m':'0','Let\'s keep it simple and say "male"':'0','M':'0','Male ':'0','Man':'0','Make':'0','Sex is male':'0','Male.':'0','Dude':'0','male ':'0',
                                         'Female':'1','female':'1','F':'1','Woman':'1','f':'1','Female ':'1','genderqueer woman':'1','fm':'1','woman':'1','female/woman':'1','female ':'1','fem':'1','Female (props for making this a freeform field, though)':'1',
                                         'Cishet male':'2','Nonbinary':'2','agender':'2','Questioning':'2','Cis Male':'2','cis woman':'2','Agender trans woman':'2','Trans man':'2','None':'2','Trans non-binary/genderfluid':'2','CIS Male':'2','Female (cis)':'2','Agender':'2','-1':'2',
                                         'Female or Multi-Gender Femme':'2','Cis male':'2','Male (cis)':'2','Other':'2','Cisgender Female':'2','none of your business':'2','genderqueer':'2','cis male':'2','Human':'2','Genderfluid':'2',
                                         'mail':'2','non-binary':'2','Male/genderqueer':'2','Cis-woman':'2','Genderqueer':'2','cisdude':'2','Genderflux demi-girl':'2','cis man':'2','Non-binary and gender fluid':'2'}})
        df_us = df_us.replace({'MHD_identification_effect_on_career':{'Yes, I think it would':'10','Yes, it has':'10','Maybe':'5','No,it has not':'0','No, it has not':'0','No, I don\'t think it would':'0'}})
        df_us = df_us.replace({'Work_remotely':{'Sometimes':'1','Never':'0','Always':'2'}})
        df_us = df_us.replace({'MHD_identification_team_reaction':{'Yes, I think they would':'10','Yes, they do':'10','Maybe':'5','No,they do not':'0','No, they do not':'0','No, I don\'t think they would':'0'}})


        #fixing the nan values
        nan_to_min1 = ['Company_size', 'Medical_leave_MH', 'Have_MHD', 'Past_MHD', 'Treatment_sought', 'Family_history_MHD',
                       'Discouranged_in_revealing_MHD_at_work',
                       'MHD_identification_effect_on_career', 'MHD_identification_team_reaction',
                       'Age', 'Gender', 'Work_remotely', 'Work_interference_with_Treatment', 'Work_interference_without_Treatment']
        nan_to_1 = ['Superviser_comfortable_talking', 'Coworkers_comfortable_talking']
        nan_to_2 = ['MentalH_insurance', 'MentalH_company_info', 'MentalH_company_resources','Anonymity_protection']
        drop_nan = []

        df_us = self.fixNan(nan_to_min1,df_us,'-1')
        df_us = self.fixNan(nan_to_1,df_us,'1')
        df_us = self.fixNan(nan_to_2,df_us,'2')

        for icol in drop_nan:
            df_us = df_us[df_us[icol].notna()]

        print(df_us.head())
        print(df_us.shape)

        for icol in colNames:
            print(icol, ' - ', df_us[icol].unique())

        self.rawData = df_us

        return self.rawData

    def data_qA(self):
        self.rawData.head()
        df = self.rawData[self.rawData['Have_MHD']=='1']
        print(df.shape)
        return df

    def data_qB(self):
        df_no_pastMHD = self.rawData[self.rawData['Past_MHD'].isin(['0','2'])]
        df = df_no_pastMHD[df_no_pastMHD['Family_history_MHD'].isin(['0','2'])]
        print(df.shape)
        return df

    def plotFeatures(self,data_dframe,f1):
        print(data_dframe['Treatment_sought'])
        feat1 = str(f1)
        #separate features vs class
        df_soughtT = data_dframe[data_dframe['Treatment_sought'] == 1]
        df_not_soughtT = data_dframe[data_dframe['Treatment_sought'] == 0]
        #plot features vs class
        fig,ax = plt.subplots()
        bval,bins,patches = plt.hist(df_soughtT[feat1].astype(float),bins = 50,facecolor="purple",label='sought Treatm')
        bval,bins,patches = plt.hist(df_not_soughtT[feat1].astype(float),bins = 50,facecolor="red",label='did not sought Treatm')
        ax.legend()
        plt.xlabel(feat1,fontsize = 16)
        plt.ylabel('Number of employees',fontsize = 16)
        plt.savefig('plots_2016/plot_'+feat1+'.png')


if __name__ == '__main__':
    dataTrain = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/OSMI_2016.csv'

    dataTest = '/Users/ioana/Desktop/Documents/univfy/test_final.csv'

    '''Prepare data for analysis'''
    train = processRawData(dataTrain)
    dataProcessed = train.processData()
    dataqA = train.data_qA()
    dataqB = train.data_qB()

    featList = ['Superviser_comfortable_talking','MentalH_insurance','MentalH_company_resources']
    for i in range(len(featList)):
            train.plotFeatures(dataqA,featList[i])
            #plt.close("all")

    dataProcessed.to_csv('dataAll_2016.csv')
    dataqA.to_csv('dataqA_2016.csv')
    dataqB.to_csv('dataqB_2016.csv')

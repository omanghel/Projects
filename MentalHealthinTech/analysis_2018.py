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

        df_0 = parameters[parameters['What country do you <strong>live</strong> in?'] == 'United States of America']
        #mapping bools to 0/1
        df_0 *= 1
        print('in US: ', df_0.shape)
        print(df_0.head())

        df_1 = df_0[df_0['<strong>Are you self-employed?</strong>'] == 0]
        print('in US, company: ', df_1.shape)
        df_us = df_1[df_1['Is your employer primarily a tech company/organization?'] == 1]
        #df_us = df_2[df_2['Is your primary role within your company related to tech/IT?'] == 1]
        print('in US, company, tech role: ', df_us.shape)

        keepCol = ['How many employees does your company or organization have?',
                  'Is your employer primarily a tech company/organization?',
                  'Does your employer provide mental health benefits as part of healthcare coverage?',
                  'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
                  'Does your employer offer resources to learn more about mental health disorders and options for seeking help?',
                  'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
                  'If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?',
                  'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?',
                  'Have you ever discussed your mental health with your employer?',
                  'Would you feel comfortable discussing a mental health issue with your coworkers?',
                  'Have you ever discussed your mental health with coworkers?',
                  'Overall, how much importance does your employer place on mental health?',
                  'Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?',
                  'Do you know local or online resources to seek help for a mental health issue?',
                  'Do you believe your productivity is ever affected by a mental health issue?',
                  'Do you currently have a mental health disorder?',
                  'Have you had a mental health disorder in the past?',
                  'Have you ever sought treatment for a mental health disorder from a mental health professional?',
                  'Do you have a family history of mental illness?',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when being treated effectively?</strong>',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when</strong> <em><strong>NOT</strong></em><strong> being treated effectively (i.e., when you are experiencing symptoms)?</strong>',
                  'Have your observations of how another individual who discussed a mental health issue made you less likely to reveal a mental health issue yourself in your current workplace?',
                  'Are you openly identified at work as a person with a mental health issue?',
                  'Has being identified as a person with a mental health issue affected your career?',
                  'If they knew you suffered from a mental health disorder, how do you think that team members/co-workers would react?',
                  'Overall, how well do you think the tech industry supports employees with mental health issues?',
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
                    'Superviser_discussion',
                    'Coworkers_comfortable_talking',
                    'Coworkers_discussion',
                    'MentalH_importance',
                    'Private_insurance',
                    'Local_resources',
                    'MentalH_productivity_effect',
                    'Have_MHD',
                    'Past_MHD',
                    'Treatment_sought',
                    'Family_history_MHD',
                    'Work_interference_with_Treatment',
                    'Work_interference_without_Treatment',
                    'Discouranged_in_revealing_MHD_at_work',
                    'MHD_identification_at_work',
                    'MHD_identification_effect_on_career',
                    'MHD_identification_team_reaction',
                    'MentalH_techIndustry_support',
                    'Age',
                    'Gender']

        df_us = df_us.rename(columns=dict(zip(keepCol,colNames)))
        print(df_us.head())
        print(df_us.shape)

        colDrop = [i for i in columnsN if not(i in keepCol)]
        print(colDrop)
        df_us = df_us.drop(columns = colDrop)
        print(df_us.head())

        icol = 'Have_MHD'
        print(df_us[icol].shape)
        print((df_us[icol].dropna()).shape)
        print(df_us[icol].unique())
        print(df_us[df_us[icol].isin(['Yes','Possibly'])].shape)

        for c in colNames:
            print(c,' - unique - ', df_us[c].unique())

        #print(df_us[icol])

        #fixing categorical variables
        df_us = df_us.replace({'Company_size':{'1-5':'1','6-25':'2','26-100':'3','100-500':'4','500-1000':'5','More than 1000':'6'}})
        df_us = df_us.replace({'MentalH_insurance':{'Yes':'1','I don\'t know':'2','No':'0','Not eligible for coverage / NA':'3'}})
        df_us = df_us.replace({'MentalH_company_info':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'MentalH_company_resources':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Anonymity_protection':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Medical_leave_MH':{'Difficult':'0','Somewhat difficult':'1','Neither easy nor difficult':'2','Somewhat easy':'3','Very easy':'4','I don\'t know':'-1'}})
        df_us = df_us.replace({'Superviser_comfortable_talking':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Coworkers_comfortable_talking':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Have_MHD':{'Yes':'1','Possibly':'1','Don\'t Know':'2','No':'0'}})
        df_us = df_us.replace({'Past_MHD':{'Yes':'1','Possibly':'1','Don\'t Know':'2','No':'0'}})
        df_us = df_us.replace({'Family_history_MHD':{'Yes':'1','I don\'t know':'2','No':'0'}})
        df_us = df_us.replace({'Work_interference_with_Treatment':{'Often':'1','Sometimes':'2','Rarely':'3','Never':'4','Not applicable to me':'-1'}})
        df_us = df_us.replace({'Work_interference_without_Treatment':{'Often':'1','Sometimes':'2','Rarely':'3','Never':'4','Not applicable to me':'-1'}})
        df_us = df_us.replace({'Discouranged_in_revealing_MHD_at_work':{'Yes':'2','Maybe':'1','No':'0'}})
        df_us = df_us.replace({'Gender':{'Male':'0','male':'0','Ostensibly Male':'0', 'm':'0','Let\'s keep it simple and say "male"':'0','M':'0','Male ':'0','Man':'0','Make':'0',
                                         'Female':'1','female':'1','F':'1','Woman':'1','f':'1','Female ':'1','woman':'1',
                                         'genderfluid':'2','Genderqueer':'2','Demiguy':'2','Trans female':'2','Cisgender male':'2','non-binary':'2','She/her/they/them':'2','Other':'2','SWM':'2','cisgender female':'2','NB':'2','Nonbinary/femme':'2','gender non-conforming woman':'2','Trans woman':'2','Cisgendered woman':'2','Cis-Female':'2','Female (cisgender)':'2','Cis woman':'2','Female/gender non-binary.':'2','non binary':'2','cis male':'2','transgender':'2','Cis-male':'2','I identify as female':'2','*shrug emoji* (F)':'2','Cishet male':'2','Nonbinary':'2','agender':'2','Questioning':'2','Cis Male':'2','cis woman':'2','Agender trans woman':'2','Trans man':'2','None':'2','Trans non-binary/genderfluid':'2','CIS Male':'2','Female (cis)':'2','Non-binary and gender fluid':'2'}})


        #fixing the nan values
        nan_to_min1 = ['Company_size', 'Medical_leave_MH', 'Superviser_discussion', 'Coworkers_discussion', 'MentalH_importance', 'Private_insurance',
                       'Local_resources', 'MentalH_productivity_effect', 'Have_MHD', 'Past_MHD', 'Treatment_sought', 'Family_history_MHD', 'Discouranged_in_revealing_MHD_at_work',
                       'MHD_identification_at_work', 'MHD_identification_effect_on_career', 'MHD_identification_team_reaction',
                       'MentalH_techIndustry_support', 'Age', 'Gender', 'Work_interference_with_Treatment', 'Work_interference_without_Treatment']
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

        for c in colNames:
            print(c,' - unique - ', df_us[c].unique())

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
        plt.savefig('plots_2018/plot_'+feat1+'.png')


if __name__ == '__main__':
    dataTrain = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/OSMI_2018.csv'

    '''Prepare data for analysis'''
    train = processRawData(dataTrain)
    dataProcessed = train.processData()

    dataqA = train.data_qA()
    dataqB = train.data_qB()

    featList = ['Superviser_discussion','MentalH_importance','MentalH_company_resources']
    for i in range(len(featList)):
            train.plotFeatures(dataqA,featList[i])
            #plt.close("all")

    dataProcessed.to_csv('dataAll_2018.csv')
    dataqA.to_csv('dataqA_2018.csv')
    dataqB.to_csv('dataqB_2018.csv')

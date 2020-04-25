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

        df_0 = parameters[parameters['What country do you *work* in?'] == 'United States of America']
        #mapping bools to 0/1
        df_0 *= 1
        print('in US: ', df_0.shape)
        print(df_0.head())

        df_1 = df_0[df_0['*Are you self-employed?*'] == 0]
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
                  'Do you *currently* have a mental health disorder?',
                  'Have you had a mental health disorder in the past?',
                  'Have you ever sought treatment for a mental health disorder from a mental health professional?',
                  'Do you have a family history of mental illness?',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work *when being treated effectively?*',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work *when* _*NOT*_* being treated effectively (i.e., when you are experiencing symptoms)?*',
                  'Have your observations of how another individual who discussed a mental health issue made you less likely to reveal a mental health issue yourself in your current workplace?',
                  'Are you openly identified at work as a person with a mental health issue?',
                  'Has being identified as a person with a mental health issue affected your career?',
                  'If they knew you suffered from a mental health disorder, how do you think that your team members/co-workers would react?',
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
        df_us = df_us.replace({'Gender':{'Male':'0','male':'0','m':'0','Let\'s keep it simple and say "male"':'0','M':'0','Male ':'0','Man':'0','Make':'0',
                                         'Female':'1','female':'1','F':'1','Woman':'1','f':'1','Female ':'1',
                                         'Cishet male':'2','Nonbinary':'2','agender':'2','Questioning':'2','Cis Male':'2','cis woman':'2','Agender trans woman':'2','Trans man':'2','None':'2','Trans non-binary/genderfluid':'2','CIS Male':'2','Female (cis)':'2','Non-binary and gender fluid':'2'}})


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

        '''
        df_us = self.fixNan('Company_size',df_us,'-1')
        df_us = self.fixNan('MentalH_insurance',df_us,'2')
        df_us = self.fixNan('MentalH_company_info',df_us,'2')
        df_us = self.fixNan('MentalH_company_resources',df_us,'2')
        df_us = self.fixNan('Anonymity_protection',df_us,'2')
        df_us = self.fixNan('Medical_leave_MH',df_us,'-1')
        df_us = self.fixNan('Superviser_comfortable_talking',df_us,'1')
        df_us = self.fixNan('Superviser_discussion',df_us,'-1')
        df_us = self.fixNan('Coworkers_comfortable_talking',df_us,'1')
        df_us = self.fixNan('Coworkers_discussion',df_us,'-1')
        df_us = self.fixNan('MentalH_importance',df_us,'-1')
        df_us = self.fixNan('Private_insurance',df_us,'-1')
        df_us = self.fixNan('Local_resources',df_us,'-1')
        df_us = self.fixNan('MentalH_productivity_effect',df_us,'-1')
        df_us = self.fixNan('Have_MHD',df_us,'-1')
        df_us = self.fixNan('Past_MHD',df_us,'-1')
        df_us = self.fixNan('Treatment_sought',df_us,'-1')
        df_us = self.fixNan('Family_history_MHD',df_us,'-1')
        #df_us = self.fixNan('Work_interference_with_Treatment',df_us,'-1')
        #df_us = self.fixNan('Work_interference_without_Treatment',df_us,'-1')
        df_us = self.fixNan('Discouranged_in_revealing_MHD_at_work',df_us,'-1')
        df_us = self.fixNan('MHD_identification_at_work',df_us,'-1')
        df_us = self.fixNan('MHD_identification_effect_on_career',df_us,'-1')
        df_us = self.fixNan('MHD_identification_team_reaction',df_us,'-1')
        df_us = self.fixNan('MentalH_techIndustry_support',df_us,'-1')
        df_us = self.fixNan('Age',df_us,'-1')
        df_us = self.fixNan('Gender',df_us,'-1')
        df_us = df_us[df_us['Work_interference_with_Treatment'].notna()]
        df_us = df_us[df_us['Work_interference_without_Treatment'].notna()]
        '''


        print(df_us.head())
        print(df_us.shape)

        '''
        comment = []
        comment.append('Describe the conversation you had with your employer about your mental health, including their reactions and what actions were taken to address your mental health issue/questions.')
        comment.append('Describe the conversation with coworkers you had about your mental health including their reactions.')
        comment.append('Describe the conversation your coworker had with you about their mental health (please do not use names).')
        comment.append('Describe the conversation you had with your previous employer about your mental health, including their reactions and actions taken to address your mental health issue/questions.')
        comment.append('Describe the conversation you had with your previous coworkers about your mental health including their reactions.')
        comment.append('Describe the conversation your coworker had with you about their mental health (please do not use names)..1')
        comment.append('Why or why not?')
        comment.append('Why or why not?.1')
        comment.append('Describe the circumstances of the supportive or well handled response.')
        comment.append('Briefly describe what you think the industry as a whole and/or employers could do to improve mental health support for employees.')
        comment.append('If there is anything else you would like to tell us that has not been covered by the survey questions, please use this space to do so.')

        print(len(df_us[comment[6]].dropna()))
        for comm in df_us[comment[6]]:
            print(comm)

        #question = 'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'
        #question = 'Do you believe your productivity is ever affected by a mental health issue?'
        #question = 'If you have a mental health disorder, how often do you feel that it interferes with your work *when being treated effectively?*'
        #question = 'If you have a mental health disorder, how often do you feel that it interferes with your work *when* _*NOT*_* being treated effectively (i.e., when you are experiencing symptoms)?*'
        question1 = 'Do you *currently* have a mental health disorder?'
        #question = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'
        question2 = 'Do you have a family history of mental illness?'
        question3 = 'Have you had a mental health disorder in the past?'
        #for comm in df_us[question2]:
        #    print(comm)
        #print(len(df_us[question].dropna()))
        #print(df_us[df_us[question].isin(['Sometimes','Often'])].shape)
        df_us_md = df_us[df_us[question1].isin(['Yes','Possibly'])]
        df_us_md_pmd = df_us_md[df_us_md[question3].isin(['Yes','Possibly'])]
        df_us_md_nofm = df_us_md[df_us_md[question2].isin(['No'])]
        print('current MD, past MD, FH = ',df_us_md_pmd[df_us_md_pmd[question2].isin(['Yes','Possibly'])].shape)
        print('current MD, past MD, no FM = ',df_us_md_pmd[df_us_md_pmd[question2].isin(['No'])].shape)
        print('current MD, no past MD = ',df_us_md[df_us_md[question3].isin(['No'])].shape)
        print('current MD, no past MD, no FM = ',df_us_md_nofm[df_us_md_nofm[question3].isin(['No'])].shape)



        maleN = ['Male','male','M','m']
        femaleN = ['Female','female','F','f']
        df_us_men = df_us[df_us['What is your gender?'].isin(maleN)]
        df_us_women = df_us[df_us['What is your gender?'].isin(femaleN)]
        print(df_us_men.shape)
        print(df_us_women.shape)

        '''

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
        plt.savefig('plots_2019/plot_'+feat1+'.png')


if __name__ == '__main__':
    dataTrain = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/OSMI_2019.csv'
    #dataTrain = '/Users/ioana/Desktop/DS4A_workspace/project_mentalHealth/OSMI_2018.csv'
    dataTest = '/Users/ioana/Desktop/Documents/univfy/test_final.csv'

    '''Prepare data for analysis'''
    train = processRawData(dataTrain)
    dataProcessed = train.processData()
    dataqA = train.data_qA()
    dataqB = train.data_qB()

    featList = ['Superviser_discussion','MentalH_importance','MentalH_company_resources']
    for i in range(len(featList)):
            train.plotFeatures(dataqA,featList[i])
            #plt.close("all")

    dataProcessed.to_csv('dataAll_2019.csv')
    dataqA.to_csv('dataqA_2019.csv')
    dataqB.to_csv('dataqB_2019.csv')

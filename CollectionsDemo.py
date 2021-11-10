from pandas.core.base import PandasObject
from pandas.core.dtypes.missing import isna, isnull
from pandas.io.parsers import read_csv
import streamlit as st 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pickle 



import warnings 
warnings.filterwarnings("ignore")

#title
st.title('Early Warning Indicator')
image=Image.open('LinkedIn Cover.jpg')
st.image(image,use_column_width=True)


def main():
    activites=['Upload Training Data','Pre-processing','Model Training','Test your Dataset']
    option=st.sidebar.radio('Select The Operation:',activites)


    if option=='Pre-processing':
        st.subheader('Exploratory Data Analysis')
        data=st.file_uploader('Upload The Dataset',type=['csv'])
        if data is not None:
            st.success('Data Successfully Uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Data:',df.head(20))
            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display Columns'):
                st.write(df.columns)
            if st.checkbox('Select Multiple Columns'):
                selected_columns=st.multiselect('Select Preferred Columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Summary'):
                st.write(df.describe().T)
            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())
#             if st.checkbox('Display The Data Type'):
#                 st.write(df.dtypes)
            if st.checkbox('Display Correlation'):
                st.write(df.corr())



    if option=='Upload Training Data':
        st.subheader('Upload Training Data')
        data=st.file_uploader('Upload The Dataset',type=['csv'])

        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Data:',df.head(20))
#             if st.checkbox('Select Columns To Plot'):
#                 selected_columns=st.multiselect('Select Your Preferred Columns',df.columns)
#                 df1=df[selected_columns]
#                 st.dataframe(df)
            
#             if st.checkbox('Display Histogram'):
#                 st.write(sns.histplot(data=df))
#                 st.pyplot()
                
#             if st.checkbox('Display Pie-Chart'):
#                 all_columns=df.columns.to_list()
#                 pie_columns=st.selectbox('Select Columns To Display',all_columns)
#                 pie_chart=df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
#                 st.write(pie_chart)
#                 st.pyplot()



    if option=='Model Training':
        st.subheader('Model Training')

        data=st.file_uploader('Upload The Dataset',type=['csv'])
        if data is not None:
            st.success('Data Successfully Uploaded')

        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Train Data:',df.head(20))
            #df.drop('Id',inplace=True, axis=1)
            df.drop('Months since last delinquent',inplace=True, axis=1)
            df.drop('Number of Credit Problems',inplace=True, axis=1)
            df.drop('Account Age',inplace=True, axis=1)
            df.drop('Customer Name',inplace=True, axis=1)
            df.drop('Agent Name',inplace=True, axis=1)
            df.drop('Purpose',inplace=True, axis=1)
            df.drop('Branch',inplace=True, axis=1)
            df.drop('Zone',inplace=True, axis=1)
            df.drop('Unnamed: 29',inplace=True, axis=1)
            df.drop('Unnamed: 32',inplace=True, axis=1)
            df1=df.dropna(subset=['Annual Income','Years in current job','Bankruptcies','Credit Score'])
            data_types_dict=dict(df1.dtypes)
            #keep track mapping column name to Label Encoder
            label_encoder_collection= {}
            for col_name, data_type in data_types_dict.items():
                if data_type=='object':
                    le=LabelEncoder()
                    df1[col_name]=le.fit_transform(df1[col_name])
                    label_encoder_collection[col_name]=le

            new_df = df1
            sc = MinMaxScaler()
            a = sc.fit_transform(df1[['Annual Income']])
            b = sc.fit_transform(df1[['Years of Credit History']])
            c = sc.fit_transform(df1[['Maximum Open Credit']])
            d = sc.fit_transform(df1[['Current Loan Amount']])
            e = sc.fit_transform(df1[['Current Credit Balance']])
            f = sc.fit_transform(df1[['Monthly Debt']])
            g = sc.fit_transform(df1[['Credit Score']])
            h = sc.fit_transform(df1[['Interest Rate']])
            new_df['Annual Income'] = a
            new_df['Years of Credit History'] = b
            new_df['Maximum Open Credit'] = c
            new_df['Current Loan Amount'] = d
            new_df['Current Credit Balance'] = e
            new_df['Monthly Debt'] = f
            new_df['Credit Score'] = g
            new_df['Interest Rate'] = h
            #st.write(new_df.head())
            #d=pd.read_csv('C:/Users/mehul/Downloads/Projects/Office Workspace/prod_data.csv')
            #st.dataframe(d.head())
            

#             if st.checkbox('Select Multiple Columns'):
#                 new_data=st.multiselect('Select Your Preferred Columns',new_df.columns)
#                 df2=new_df[new_data]
#                 st.dataframe(df2)

            x=new_df.iloc[:,:-1]
            y=new_df.iloc[:,-1]
            #seed=st.sidebar.slider('Random States',1,200)
            classifier_name=st.sidebar.selectbox('Select Your Preferred Model',('XGBOOST','KNN'))

            def add_parameter(name_of_clf):
                param=dict()
                
                if name_of_clf=='KNN':
                    K=st.sidebar.slider('K',1,15)
                    param['K']=K
                    return param
            param=add_parameter(classifier_name)


            def get_classifier(name_of_clf,param):
                clf= None
                if name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=param['K'])
                elif name_of_clf=='XGBOOST':
                    clf=XGBClassifier()
                else:
                    st.warning('Select Your Preferred Algorithm')
                return clf
            
            clf=get_classifier(classifier_name,param)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
            clf.fit(x_train,y_train)

            y_pred=clf.predict(x_test)
#             st.write(y_pred)

            

            pickle_out = open("classifier.pkl", mode = "wb") 
            pickle.dump(clf, pickle_out) 
            pickle_out.close()

            accuracy=accuracy_score(y_test,y_pred)
            score = np.round(accuracy * 100)
            st.write('Name Of Classifier:',classifier_name)
            if st.button('Train'):
                st.write('Model Training Accuracy Score in (%) is:',score)

    if option=='Test your Dataset':
      st.subheader('Predict defaults')
    #   activites_test=['Explore Outcome','Look-a-like Analysis']
    #   option_test=st.sidebar.radio('Select The Operation:',activites_test)          
      test_data = st.file_uploader('Upload Test Dataset',type=['csv'])
      if test_data is not None:
            test_df=pd.read_csv(test_data)
            st.write('Raw Data:',test_df)
            new_test_df = test_df.copy()
            new_test_df.drop('Months since last delinquent',inplace=True, axis=1)
            new_test_df.drop('Number of Credit Problems',inplace=True, axis=1)
            new_test_df.drop('Account Age',inplace=True, axis=1)
            new_test_df.drop('Customer Name',inplace=True, axis=1)
            new_test_df.drop('Agent Name',inplace=True, axis=1)
            new_test_df.drop('Purpose',inplace=True, axis=1)
            new_test_df.drop('Branch',inplace=True, axis=1)
            new_test_df.drop('Zone',inplace=True, axis=1)
            new_test_df.drop('Unnamed: 35',inplace=True, axis=1)
            #est_df.drop('Unnamed: 32',inplace=True, axis=1)
            #df1=df.dropna(subset=['Annual Income','Years in current job','Bankruptcies','Credit Score'])
            new_test_df1=new_test_df.dropna(subset=['Annual Income','Years in current job','Bankruptcies','Credit Score'])
            data_types_dict=dict(new_test_df1.dtypes)
            #keep track mapping column name to Label Encoder
            label_encoder_collection= {}
            for col_name, data_type in data_types_dict.items():
                if data_type=='object':
                    le=LabelEncoder()
                    new_test_df1[col_name]=le.fit_transform(new_test_df1[col_name])
                    label_encoder_collection[col_name]=le

            new_df1 = new_test_df1
#             st.write(new_df1)
            sc = MinMaxScaler()
            a1 = sc.fit_transform(new_test_df1[['Annual Income']])
            b1 = sc.fit_transform(new_test_df1[['Years of Credit History']])
            c1 = sc.fit_transform(new_test_df1[['Maximum Open Credit']])
            d1 = sc.fit_transform(new_test_df1[['Current Loan Amount']])
            e1 = sc.fit_transform(new_test_df1[['Current Credit Balance']])
            f1= sc.fit_transform(new_test_df1[['Monthly Debt']])
            g1 = sc.fit_transform(new_test_df1[['Credit Score']])
            h1 = sc.fit_transform(new_test_df1[['Interest Rate']])
            new_df1['Annual Income'] = a1
            new_df1['Years of Credit History'] = b1
            new_df1['Maximum Open Credit'] = c1
            new_df1['Current Loan Amount'] = d1
            new_df1['Current Credit Balance'] = e1
            new_df1['Monthly Debt'] = f1
            new_df1['Credit Score'] = g1
            new_df1['Interest Rate'] = h1
            #st.write(new_df1.head())
            new_df_lk = new_df1
            
            #st.write(new_df_lk)
            
#             st.write(new_df1)

            # loading the trained model
            pickle_in = open('classifier.pkl', 'rb') 
            classifier = pickle.load(pickle_in)
            
            test_predict = classifier.predict(new_df1)
#             st.write(test_predict) 
            test_defaults = pd.DataFrame(test_predict)
            test_defaults.rename(columns = {0:'Risk_Category'}, inplace = True)
           
            # st.write(plk1)
            p=test_defaults['Risk_Category'].replace({0:'Non-defaulter', 1:'Defaulter'})
            #st.write(p.head())
            
  
            probabilities = classifier.predict_proba(new_df1)
            q = pd.DataFrame(probabilities)
            # #st.write(q.head())


            q.rename(columns = {0:'Non-defaulter Prob', 1:'Risk_Score'}, inplace = True)
            q.drop('Non-defaulter Prob',inplace=True, axis=1)
            #q = q.round()
            #q['Risk_Score'].round(decimals = 2)
            # q['Risk_Score'].round(decimals=2)
            q['Risk_Score'] = q['Risk_Score'] * 100
            # q.sort_values(q['Risk_Score'],ascending=False)
            # # q = q.round(2)
#             st.write(q)
            
            def f(row):
                if row['Risk_Score'] <29:
                    val= 'Low Risk'
                elif row['Risk_Score'] <70:
                    val= 'Medium Risk'
                else:
                    val=  'High Risk'
                return val
    #             s = pd.DataFrame(r)
            def g(row):
                return np.round_(row['Risk_Score'], decimals=2)
            
            q["Risk Category"]=q.apply( f,axis=1)
            q["Risk_Score"]=q.apply( g,axis=1)
#             st.write(q)

            # result_lk = [plk,q]
            # result_lk = pd.concat([plk1,q], axis=1)
            #result_lk = pd.DataFrame(result_lk)

            # df_lk = [new_df_lk,result_lk]
            # df_lk = pd.concat([new_df_lk,result_lk], axis = 1)
            # st.write(df_lk)
            
            result=[p,q]
            result = pd.concat([p, q], axis=1)
            #results = pd.concat(result)
            #st.write(result.head())
            import math
            output=[test_data,result]
            output_df = pd.concat([test_df, result], axis=1)
            output_df.drop('Unnamed: 35',inplace=True, axis=1)
            output_df.drop('Risk_Category',inplace=True, axis=1)
            output_df.sort_values(by=['Risk_Score'], inplace=True, ascending=False)
            output_df['Risk_Score']=pd.to_numeric(output_df['Risk_Score'])
            output_df['Risk_Score'] = output_df['Risk_Score'].round(decimals=2)
            output_df=output_df.dropna(subset=['Risk_Score','Risk Category'])
            st.write('Prediction on the Test Data:',output_df)

           

    #st.write(outcome_df)
    #   if option=='Explore Outcome':
    #     st.subheader('Filterings')
            fil = st.checkbox('Explore Outcome')
            if fil == True:
                filter_df = output_df
            
      #filter_df = pd.read_csv('C:/Users/mehul/Downloads/Projects/Office Workspace/asd.csv')

                filter_levels = ['Branch','Zone','Agent Name']
                choice = st.selectbox('Select the Level:', filter_levels)

                filter_metrics = ['Top 10','Bottom 10','All']
                ch1=st.selectbox('Select the Metric:', filter_metrics)


                if choice == 'Branch':
            #st.write(outcome.head())

                    filter_data = st.text_input('Enter the Branch Name:')
                    filtered=(filter_df[filter_df['Branch'] == filter_data])
                    

                    st.write('No of Customers by Risk Category:',filtered['Risk Category'].value_counts())  
                    st.write('Collection Outstanding: Rs.', filtered['Maximum Open Credit'].sum())
                    st.write("Customers filtered by ", filter_data)
                    st.write(filtered)
                    
                    top_branch = (filtered.sort_values('Risk_Score', ascending=False))
                             
                    if ch1 == 'Top 10':
                        top_branch1= top_branch[top_branch['Risk Category'] != 'Low Risk']
                        st.write('High Risk Customers- Top 10:',top_branch1.head(10))

                    elif ch1 == 'Bottom 10':
                        
                        st.write('Low Risk Customers- Bottom 10:',top_branch.tail(10))

                    elif ch1 == 'All':
                        st.write('All :',top_branch)
                    else:
                        st.write('Please select the Metric')


                elif choice == 'Zone':

                    g=filter_df.groupby(['Zone','Risk Category']).size().reset_index(name = 'counts')
                    st.write('Zone Wise Performance:', g)
                    #g.sort_values(by = ['counts'], axis = 0, ascending = False, inplace = True)
                    # gb = pd.DataFrame(g.first())
                    # gb.columns = ['Zone','Risk Category']
                    hr = g[g['Risk Category'] == 'High Risk']
                    hr.sort_values(by = ['counts'], axis = 0, ascending = False, inplace = True)
                    st.write('High Risk Zones:', hr)
                
                    filter_zone = st.text_input('Enter the Zone Name:')
                    filtered_zone =(filter_df[filter_df['Zone'] == filter_zone])
                    
                    st.write('No of Customers by Risk Category',filtered_zone['Risk Category'].value_counts())
                    st.write('Collection Outstanding: Rs. ', filtered_zone['Maximum Open Credit'].sum())
                    st.write("Customers filtered by Zone",filter_zone )
                    st.write(filtered_zone)

                    
                    top_zone = (filtered_zone.sort_values('Risk_Score', ascending=False))
                    

                    if ch1 == 'Top 10':
                        top_zone1= top_zone[top_zone['Risk Category'] != 'Low Risk']
                        st.write('High Risk Customers- Top 10:',top_zone1.head(10))

                        # top_zone2= g[g['Risk Category'] .value_counts()]
                        # st.write('High Risk Zones- Top 10:',top_zone1.head(10))


                    elif ch1 == 'Bottom 10':
                        st.write('Low Risk Customers- Bottom 10:',top_zone.tail(10))

                    elif ch1 == 'All':
                        st.write('All :',top_zone)
                    else:
                        st.write('Please select the Metric')


                elif choice == 'Agent Name':
                    filter_agent = st.text_input('Enter the Agent Name:')
                    filtered_agent=(filter_df[filter_df['Agent Name'] == filter_agent])
                    
                    st.write('No of Customers by Risk Category',filtered_agent['Risk Category'].value_counts())  
                    st.write('Collection Outstanding: Rs.', filtered_agent['Maximum Open Credit'].sum())
                    st.write("Customers filtered by Agent name: ", filter_agent)
                    st.write(filtered_agent)     
                    top_agent = (filtered_agent.sort_values('Risk_Score', ascending=False))

                    if ch1 == 'Top 10':
                        top_agent1= top_agent[top_agent['Risk Category'] != 'Low Risk']

                        st.write('High Risk Customers- Top 10:',top_agent1.head(10))

                    elif ch1 == 'Bottom 10':
                        st.write('Low Risk Customers- Bottom 10:',top_agent.tail(10))

                    elif ch1 == 'All':
                        st.write('All:',top_agent)
                    else:
                        st.write('Please select the Metric')

                else:
                    st.write('Please, Select the Level')            
            





            fil1 = st.checkbox('Look-a-Like Analysis')
            if fil1 == True:
                clean_df = output_df.copy()
                test_lk = output_df.copy()
                #st.write(output_df.shape)
                
                clean_df.drop('Customer Name', inplace = True, axis=1)
                clean_df.drop('Agent Name', inplace = True, axis=1)
                clean_df.drop('Branch', inplace = True, axis=1)
                clean_df.drop('Zone', inplace = True, axis=1)
                clean_df.drop('Months since last delinquent',inplace=True, axis=1)
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('< 1 year','0.5')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('years','')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('year','')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('+','')
                clean_df=clean_df.dropna(subset=['Years in current job'])
                #clean_df = clean_df['Years in current job'].apply(pd.to_numeric, errors='coerce')
                #st.write(clean_df['Years in current job'].dtypes)
                #clean_df=clean_df.fillna(['Years in current job'].mean(), inplace=True)
                #st.write(clean_df)

                data_types_dict=dict(clean_df.dtypes)
                label_encoder_collection= {}
                for col_name, data_type in data_types_dict.items():
                    if data_type=='object':
                        le=LabelEncoder()
                        clean_df[col_name]=le.fit_transform(clean_df[col_name])
                        label_encoder_collection[col_name]=le

                #st.write(clean_df.shape)
                
                clean_df1 = clean_df
                #st.write(clean_df1)
                sc = MinMaxScaler()
                # a1 = sc.fit_transform(new_test_df1[['Annual Income']])
                # b1 = sc.fit_transform(new_test_df1[['Years of Credit History']])
                # c1 = sc.fit_transform(new_test_df1[['Maximum Open Credit']])
                # d1 = sc.fit_transform(new_test_df1[['Current Loan Amount']])
                # e1 = sc.fit_transform(new_test_df1[['Current Credit Balance']])
                # f1= sc.fit_transform(new_test_df1[['Monthly Debt']])
                # g1 = sc.fit_transform(new_test_df1[['Credit Score']])
                # h1 = sc.fit_transform(new_test_df1[['Interest Rate']])
                clean_df1['Annual Income'] =  sc.fit_transform(clean_df[['Annual Income']])
                clean_df1['Years of Credit History'] = sc.fit_transform(clean_df[['Years of Credit History']])
                clean_df1['Maximum Open Credit'] = sc.fit_transform(clean_df[['Maximum Open Credit']])
                clean_df1['Current Loan Amount'] = sc.fit_transform(clean_df[['Current Loan Amount']])
                clean_df1['Current Credit Balance'] = sc.fit_transform(clean_df[['Current Credit Balance']])
                clean_df1['Monthly Debt'] = sc.fit_transform(clean_df[['Monthly Debt']])
                clean_df1['Credit Score'] = sc.fit_transform(clean_df[['Credit Score']])
                clean_df1['Interest Rate'] = sc.fit_transform(clean_df[['Interest Rate']])

                #st.write(clean_df1.shape)

                scaled_df = clean_df1
                #st.write(scaled_df)
                scaled_df.drop('Id', axis = 1)
                nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(scaled_df)
                distances, indices = nbrs.kneighbors(scaled_df)
                dist_df = pd.DataFrame(distances)
                dist_df.reset_index(inplace = True, drop = True)
                #st.write(dist_df.head())
                dist = dist_df * 1000
                # dist_ = pd.to_numeric(dist)
                # dist_= dist.round(decimals=2)


            
                #st.write(dist)
                
                ind_df=pd.DataFrame(indices)
                ind_df.reset_index(inplace = True, drop = True)
                #st.write(test_lk)
                #st.write(ind_df)
                lk_output_ind=pd.concat([test_lk,ind_df], axis=1)


                lk_output=pd.concat([test_lk,ind_df], axis=1)
                lk_output.reset_index(inplace = True, drop = True)
                st.write('Customer Information with Similarity Scores: ', lk_output)



                Customer_id1 = st.number_input('Enter Customer Id:', step = 1)
                user_value=Customer_id1
                # user_value=Customer_id1
                temp3=pd.DataFrame()
                for i in range(0,11):
                    temp=lk_output_ind[[0,1,2,3,4,5,6,7,8,9,10]].iloc[user_value,i]
                    temp2=pd.DataFrame(lk_output_ind.loc[lk_output_ind.index==temp])
                    temp2.drop('Id', inplace=True, axis=1)
                    temp3=temp3.append(temp2)
                    #st.write(temp2)
                st.write('Top 10 Similar Customers:',temp3)
                
                
                # # a = 1
                # # if a!=notnull:
                # #     def1=0
                # a=st.text_input('Enter Customer ID:')
                # # else:
                # #     a = st.text_input('Enter Customer ID:')

                # b = pd.to_numeric(a)
                #     #b.dtype
                # st.write(a)
                #     #Id=4
                # temp2=pd.DataFrame()
                # for i in range(0,11):
                #     temp=lk_output[[0,1,2,3,4,5,6,7,8,9,10]].iloc[b,i]
                #     temp2 = pd.DataFrame(lk_output.loc[lk_output.index==temp])
                #     #temp2.drop([0,1,2,3,4,5,6,7,8,9,10],inplace=True,axis=1)
                #     (temp2)

                # a = st.text_input('Enter Customer ID:')
                # b = pd.to_numeric(a)
                #     #b.dtype
                # st.write(a)
                #     #Id=4
                # temp2=pd.DataFrame()
                # for i in range(0,11):
                #     temp=lk_output[[0,1,2,3,4,5,6,7,8,9,10]].iloc[b,i]
                #     temp2 = pd.DataFrame(lk_output.loc[lk_output.index==temp])
                #     temp2.drop([0,1,2,3,4,5,6,7,8,9,10],inplace=True,axis=1)
                #     (temp2)







#             fil1 = st.checkbox('Look-a-Like Analysis')
#             if fil1 == True:
#                 new_df_lk.dropna(inplace=True)
#                 dataset1reference=test_df.reset_index(inplace = True, drop = True)
# #                 st.write(test_df)
#                 nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(new_df_lk)    
#                 distances, indices = nbrs.kneighbors(new_df_lk)
#                     #nbrs.kneighbors

#                     #indices
#                 ind_df=pd.DataFrame(indices)
# #                 ind_df.reset_index(inplace = True, drop = True)

#                 lk_output=pd.concat([dataset1reference,ind_df], axis=1)
#                     # lk_output_df = st.dataframe(lk_output)

#                 a = st.number_input('Enter Customer ID:',step =1)
#                 b = pd.to_numeric(a)
#                     #b.dtype
#                 st.write(lk_output)
#                     #Id=4
#                 temp2=pd.DataFrame()
#                 for i in range(0,11):
#                     temp=lk_output[[0,1,2,3,4,5,6,7,8,9,10]].iloc[b,i]
#                     temp2 = pd.DataFrame(lk_output.loc[lk_output.index==temp])
#                     temp2.drop([0,1,2,3,4,5,6,7,8,9,10],inplace=True,axis=1)
#                     (temp2)
#                 # nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(new_df_lk)    
#                 # distances, indices = nbrs.kneighbors(new_df_lk)
#                 #     #nbrs.kneighbors

#                 #st.write(distances)
#                 # Customer_id1=st.number_input("Please Enter Customer ID",step=1)
#                 # look= pd.DataFrame(indices[int(Customer_id1)])
#                 # look.columns=["index"]
#                 #st.write("Top 10 Most similar Customers",look)
                
#                 # alike = test_df.iloc[:,0:2]
#                 # ref = pd.DataFrame(alike)
#                 # #st.write(ref)

#                 # ind_df=pd.DataFrame(distances)
#                 # ind_df.reset_index(inplace = True, drop = True)
#                 # lk_output=pd.concat([ref,ind_df], axis=1)
#                 # #lk_output_df = st.write('Customer Details with Similarity Scores:',lk_output)
                
#                 # ind_df1=pd.DataFrame(indices)
#                 # ind_df1.reset_index(inplace = True, drop = True)
#                 # lk_output1=pd.concat([new_df_lk,ind_df], axis=1)
#                 # #lk_output_df1 = st.write('Customer Details with Similarity Scores:',lk_output)

                
#                 # if z == Null:
#                 #      def1 = 0
#                 #      z = st.text_input('Enter Customer ID:',def1)
#                 # else:
#                 # z = st.text('Enter Customer ID:')
                
#                 # b = pd.to_numeric(z)
#                 #     #b.dtype
#                 # st.write(z)
#                 #     #Id=4
#                 # temp2=pd.DataFrame()
#                 # for i in range(0,11):
#                 #     temp=lk_output1[[0,1,2,3,4,5,6,7,8,9,10]].iloc[b,i]
#                 #     temp2 = pd.DataFrame(lk_output1.loc[lk_output1.index==temp])
#                 #     temp2.drop([0,1,2,3,4,5,6,7,8,9,10],inplace=True,axis=1)
#                 #     temp3 = pd.DataFrame(temp2)
#                 #     st.write(temp3)

   


if __name__ == '__main__':
    main()


                
                

        
        
    
    
    

    

            
                



                    

               

                
            
        


            


import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import pickle


def to_excel(df):
    """
    Accepts a dataframe and outputs an excel file.

    """

    output = BytesIO()
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',index=False)
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded.
    in:  dataframe
    out: href string

    """
    val = to_excel(df)              #Getting the dataframe in an excel
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="churn_template.csv">Download template for uploading</a>' # decode b'abc' => abc

def uploader():
    """

    This function creates the skeleton dataframe to be converted into excel, receives the excel from the user and sends it back to __main__

    'vals' indicates acceptable column entries
    'd' is the skeleton dataframe

    """
    vals = [0,'Male/Female','Yes/No','Yes/No','Yes/No',123,'Yes/No','Yes/No/No phone service','DSL/Fibre optic/No','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Month-to-month/One year/Two year','Yes/No','Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)',123,123]
    d= pd.Series(vals, index=['CustomerID','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges'])
    d=pd.DataFrame(d).transpose()
    
    st.markdown(get_table_download_link(d), unsafe_allow_html=True)
    uploaded_file=st.file_uploader("Please upload data in the provided format")
    
    if uploaded_file is not None:
        dff=pd.read_excel(uploaded_file)
        dff=dff[dff.CustomerID!=0]
        return dff

    else:
        st.markdown("Waiting for your input...(You can find a filled template at my [github](https://github.com/coderkol95/Telecom_churn) repo.)",unsafe_allow_html=True)    

###############################################################################
# __main__():
#
# 1. Wait till the user uploads the excel and its received in a dataframe
# 2. Store the customer ID
# 3. Convert seniorcitizen column to string if not 
# 4. Generate new features based on already fitted pipeline
# 5. Predict the output on the basis of already fitted model!
# 6. Concatenate the output with the previously stored customer ID
# 7. Print the result to the console

###############################################################################

if __name__=='__main__':
  
    st.title("Phone company's churn prediction module for the Martian Sapiens")

    df=uploader()
    if df is not None:
        
        custID=df['CustomerID']
        df.drop(['CustomerID'],axis=1,inplace=True)
        
        df.SeniorCitizen=df.SeniorCitizen.apply(lambda x: str(x))       
        
        with open (r'./bin/preprocessing.pkl','rb') as r:
            preprocess=pickle.load(r)

        with open(r'./bin/model.pkl','rb') as a:
            model=pickle.load(a)

        model_frame=preprocess.transform(df)

        y=model.predict(model_frame)
        out=pd.DataFrame(y,index=custID)
        out.columns=['Likely to churn']
        out.index.name='Customer ID'
        out['Likely to churn'].astype('bool')
        st.write(out)


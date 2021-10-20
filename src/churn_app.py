import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import pickle
import re
import uuid
import json
import io

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """

    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        #object_to_download = object_to_download.to_csv(index=False)
        towrite = io.BytesIO()
        object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
        towrite.seek(0)

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def uploader():
    """

    This function creates the skeleton dataframe to be converted into excel, receives the excel from the user and sends it back to __main__

    'vals' indicates acceptable column entries
    'd' is the skeleton dataframe

    """
    vals = [0,'Male/Female','Yes/No','Yes/No','Yes/No',123,'Yes/No','Yes/No/No phone service','DSL/Fibre optic/No','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Yes/No/No internet sevice','Month-to-month/One year/Two year','Yes/No','Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)',123,123]
    d= pd.Series(vals, index=['CustomerID','gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure(months)','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','MonthlyCharges(Rs.)', 'TotalCharges(Rs.)'])
    d=pd.DataFrame(d).transpose()


    filename = 'churn-prediction-format.xlsx'
    download_button_str = download_button(d, filename, f'Download churn-prediction-format')
    st.markdown(download_button_str, unsafe_allow_html=True)

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
        
        try:
            custID=df['CustomerID']
            df.drop(['CustomerID'],axis=1,inplace=True)
        
            df.SeniorCitizen=df.SeniorCitizen.apply(lambda x: str(x))       


            try:
                with open (r'./bin/preprocessing.pkl','rb') as r:
                    preprocess=pickle.load(r)

                with open(r'./bin/model.pkl','rb') as a:
                    model=pickle.load(a)

                model_frame=preprocess.transform(df)
                y=model.predict(model_frame)
                
                out=pd.DataFrame(y,index=custID)
                out.columns=['Likely to churn']
                out['Likely to churn'].replace({'1':'Yes','0':'No'}, inplace=True)
                out.index.name='Customer ID'
                out['Likely to churn'].astype('bool')
                st.write(out)
                out.reset_index(inplace=True)
                
                filename = 'churn-predictions.xlsx'
                download_button_str = download_button(out, filename, f'Download churn-predictions')
                st.markdown(download_button_str, unsafe_allow_html=True)

            except:
                st.write('Internal error. Please contact Anupam.')

        except:

            st.write('The data has not been entered correctly in the template.')
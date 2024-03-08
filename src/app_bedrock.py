############################Industrial Generative AI Assistant###################
############################## First Author:Julia Hu ##################################
############################## Second Author: Sudeesh Sasidharan##################################

#########################import library#############################
import json
import sys
import boto3
import streamlit as st
import datetime as dt
import os
import re
import pandas as pd
from pandasai import SmartDataframe
from pandasai.callbacks import StdoutCallback
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from IPython.display import display_markdown,Markdown,clear_output
from requests_aws4auth import AWS4Auth
from langchain.load.dump import dumpd
from langchain.vectorstores import OpenSearchVectorSearch
from pandasai.responses.streamlit_response import StreamlitResponse
from typing import Dict
from urllib.request import urlretrieve
from CodeGenerationClass import CodeGenerationClass
from ImageProcessClass import ImageProcessClass
from ImageProcessClass import StreamProcessor
from KnowledgebaseRAG import KnowledgebaseRAGClass
###########################Global Variables##########################
# utility functions
def get_cfn_outputs(stackname: str) -> str:
    cfn = boto3.client('cloudformation','us-east-1')
    outputs = {}
    for output in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs

###########################Global Variables##########################
CFN_STACK_NAME = "genai-sagemaker"
outputs = get_cfn_outputs(CFN_STACK_NAME)
region = outputs["Region"]
get_bucket_name_from_arn = lambda s3_arn: s3_arn.split(':')[-1] if s3_arn.startswith("arn:aws:s3:::") else None
s3_arn = outputs["S3Bucket"]
s3_bucket = get_bucket_name_from_arn(s3_arn)
s3_prefix = 'monitron'

service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

def main():
    if len(sys.argv) > 1:
        knowledgebase_arn=str(sys.argv[1])
        
    else: print('Please provide knowledgebase arn as an input')
    ###############Setup Boto3 Clients####################
    s3 = boto3.client('s3')
    rekognition = boto3.client('rekognition', region_name = region)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)

    llm = Bedrock(
        client=bedrock_runtime,
        model_id="anthropic.claude-v2"
    )
    bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)

    ####################################################################################################
    #################################Opensearch Index for Pandas Code Embedding#########################
    ####################################################################################################
    indexname = "genai-monitron-nlq"
    aoss_collection_arn = outputs['CollectionARN']
    host = f"{os.path.basename(aoss_collection_arn)}.{region}.aoss.amazonaws.com:443"

    opensearch_vector_search = OpenSearchVectorSearch(
        opensearch_url=host, 
        embedding_function=bedrock_embeddings, 
        index_name=indexname,
        http_auth=auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )

    ####################################################################################################
    ######################Opensearch Index for KnowledgeBase Root Cause Analysis#########################
    ####################################################################################################
    # ENTER THE knowledgebase_arn from the OpenSearch Collection created for the knowledge base
    aoss_collection_arn = knowledgebase_arn
    aoss_host = f"{os.path.basename(aoss_collection_arn)}.{region}.aoss.amazonaws.com"
    aoss_vector_index = "bedrock-knowledge-base-default-index"
    client = OpenSearch(
        hosts = [{'host': aoss_host, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20
    )

    ###############Setup streamlite page config####################
    st.set_page_config(page_title="GenAI Monitron Dataset Analyzer", page_icon="eyeglasses")
    languages = ['English', 'Spanish', 'German', 'Portugese', 'Irish', 'Korean', 'Swedish', 'Norwegian', 'Danish', 'Icelandic', 'Finnish', 'Star Trek - Klingon', 'Star Trek - Ferengi', 'Italian', 'French', 'Japanese', 'Mandarin', 'Tamil', 'Hindi', 'Telugu', 'Kannada', 'Arabic', 'Hebrew']
    # Layout
    st.title('GenerativeAI Industrial AI Analyzer')

    st.markdown("# Simplify Monitron Data Analysis")
    st.sidebar.header("Select an output language")
    default_lang_ix = languages.index('English')
    language = st.sidebar.selectbox(
        'Only Alpha and Beta quadrant languages supported. For new requests, please contact C-3PO',
        options=languages, index=default_lang_ix)

    with st.sidebar.expander("**Click here for sample questions...**"):
                        st.markdown(
                            """
                            - Easy 😄
                                - Can you tell me all unique sites in this doc?
                                - Can you tell me all unique assets in this doc?
                                - What is the total number of unique sensors from this doc?
                                - what is the earlist timetamp and latest timestamp of this dataset?
                            - Hard 🤫
                                - Tell me the number of sensors for each site. 
                                - What is number of sensors from each site shown vibration signal in Warning or Alarm status?
                                - What is the average acceleration for sensors shown vibrationML as NOT Healthy?
                                - What is the average temperature for sensors shown temperatureML as NOT Healthy?
                                - What is number of sensors in Site C shown temperature in Warning, and Alarm status respectively?
                            
                            - Challenge 😵‍💫
                                - What percentage of sensors at each site shown vibration signal in NOT healthy status?
                                - For sensors shown vibration signal as NOT Healthy, can you calculate the time duration in days for each sensor shown abnormal vibration signals?
                                - For sensors shown temperature as NOT Healthy, can you calculate the time duration in days for each sensor shown abnormal temperatureML?
                                - What is number of sensors shown vibration in Not Healthy status at Site C monthly?
                                - What is percent change of number sensors showing vibration signal Not Healthy status for Site A monthly?
                        """
                        )

    # Initiate class for code geneeration, imagechat
    Codegeneration = CodeGenerationClass(opensearch_vector_search, 4,indexname)
    ImageChatHandler = ImageProcessClass(rekognition,bedrock_runtime)
    KnowledgebaseHandler=KnowledgebaseRAGClass(bedrock_embeddings, client, aoss_vector_index,3)     

    ###############different tabs streamlit####################
    tab1, tab2, tab3 = st.tabs(["Data Insight & Chart", "Captured Image Summary", "Root Cause Analysis"])
    with tab1:
        st.header("📈 Data Insight & Chart")
        st.subheader("Upload your csv file")
        uploaded_file = st.file_uploader("**Select a file**", type=['csv'])
        if uploaded_file is not None:
            st.success(uploaded_file.name + ' is ready for upload')
            if 'csv' in uploaded_file.name:
                load_csv = st.button('Upload', key = 'file')
                if "load_state_csv" not in st.session_state:
                    st.session_state.load_state_csv = False
                if load_csv or st.session_state.load_state_csv:
                    st.session_state.load_state_csv = True
                    s3.upload_fileobj(uploaded_file, s3_bucket, s3_prefix+'/'+uploaded_file.name)
                    st.success('File uploaded')
                    s3.download_file(s3_bucket, s3_prefix+'/'+uploaded_file.name, uploaded_file.name)
                    # You can instantiate a SmartDataframe with a path to a CSV file
                    df = SmartDataframe(pd.read_csv(uploaded_file.name), {"callback": StdoutCallback()}, {"enable_cache": False}, config={"llm": llm, "save_charts":False, "response_parser": StreamlitResponse})
                    if len(df) > 0:
                        ###########################Text box to allow end users put in questions###################################################        
                        st.write("**Chat With You IoT Data**")
                        input_text = st.text_input('**What insights would you like?**', key='text', placeholder="Type a question you want to know about your asset anomaly...")
                        if input_text != '':
                            st.write("**Your question is submited as:**", input_text)
                            refined_prompt = Codegeneration.refineprompt(input_text) 
                            st.write(df.chat(refined_prompt))
                        ###########################output last_generated_code###################################################
                        st.divider() 
                        with st.expander("**Code Generated From Your Question**"):        
                            st.code(df.last_code_executed, language='python')  
                            
                        ###########################collect human feedback###################################################
                        st.divider()
                        st.write("**User Feedback Form**")
                        human_response = st.radio("**Does this result help you?**",key="user_feedback",options=["Not Helpful 👎","Helpful 👍"])
                        if st.session_state.user_feedback=="***Helpful*** 👍":
                            st.write("Thanks!")
                        elif st.session_state.user_feedback=="***Not Helpful 👎***":
                            st.write("I will be better next time!")
                        else:
                            st.write("Please select one option above")
                        selection = st.session_state.user_feedback
                        ###########################submit feedback to opensearch###################################################
                        if selection == "Helpful 👍" and input_text !='' and df.last_code_executed != '':
                            response = Codegeneration.addtext_opensearch(input_text, str(df.last_code_executed))
                        else:
                            pass
                        st.write("**Your Feedback is submited**", selection)

            else:
                st.error('Incorrect file type provided. Please select a csv file to proceed', icon="🚨")
    #######################Use case 2 Image upload and summary################################## 
    with tab2:
        st.header("🎥 Captured Image Summary")
        st.subheader("Upload your captured image file")
        uploaded_img = st.file_uploader("**Select a file**", type=['png','jpg','jpeg'])
        if uploaded_img is not None:
            st.success(uploaded_img.name + ' is ready for upload')
            if 'jpg' in uploaded_img.name or 'png' in uploaded_img.name or 'jpeg' in uploaded_img.name:
                load_imag = st.button('Upload', key = 'image')
                if "load_state_imag" not in st.session_state:
                    st.session_state.load_state_imag = False
                if load_imag or st.session_state.load_state_imag:
                    st.session_state.load_state_imag = True
                    st.image(uploaded_img)
                    st.markdown('**Image summary**: \n')
                    stream_output = st.empty() #create a container to hold the streaming output
                    stream_processor = StreamProcessor(stream_output) #create a StreamProcessor instance with the output container  
                    with st.spinner('Uploading image file and starting summarization with Amazon Rekognition label detection...'):
                        query =ImageChatHandler.upload_image_detect_labels(uploaded_img.getvalue(), language)
                        ImageChatHandler.call_bedrock_claude(query,streaming_callback=stream_processor.streaming_handler)    
                        st.session_state['img_summary'] = stream_processor.combined_text
                        s3.upload_fileobj(uploaded_img, s3_bucket, s3_prefix+'/'+uploaded_img.name)
                        st.success('File uploaded and summary generated')
            else:
                st.error('Incorrect file type provided. Please select either a JPG or PNG file to proceed', icon="🚨")
    #########################Root Cause Analysis##########################
    with tab3:
        # 1. Start with the query
        st.header("🔧 Root Cause Analysis")
        input_text = st.text_input('**What insights would you like?**', key='root_cause', placeholder="Type a question you want to know about your asset anomaly...")
        
        if st.button("Submit"):
            # 1. Create the context by finding similar documents from the knowledge base
            relevant_documents, context = KnowledgebaseHandler.query_docs(input_text)
            
            # 2. Now reformat prompt to combine question and context together
            prompt = KnowledgebaseHandler.get_prompt(context, input_text)

            # 3. Provide the prompt to the LLM to generate an answer to the query based on context provided
            response = llm(prompt)
            # 4. Reformat answer for App: Remove <answer> from the beginning
            cleaned_response = re.sub(r'<answer>', '', response).split(" — ")
            # Remove (solution code) from each sentence
            final_response = [re.sub(r'\([^)]*\)', '', sentence).strip() for sentence in cleaned_response]
            # Combine the modified sentences into a paragraph
            combined_paragraph = " — ".join(final_response)
            
            st.write(combined_paragraph)
            # Also provide source of information
            st.divider() 
            with st.expander("**Source of Answer**"):
                for r in relevant_documents['hits']['hits']:
                    s = r['_source']
                    st.write(f"{s['AMAZON_BEDROCK_METADATA']}\n{s['AMAZON_BEDROCK_TEXT_CHUNK']}")
                    st.write("----------------")

if __name__ == "__main__":
    main()

# Author: Julia Hu
#########################import library#############################
import json
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

class CodeGenerationClass:
    """
    A class to use PandasAI generated code to improve chat accuracy
    """
    def __init__(self, opensearch_vector_search,kexamples,indexname):
        self.opensearch_vector_search = opensearch_vector_search
        self.kexamples = kexamples
        self.indexname = indexname
    
    def refineprompt(self,input_question):
        """
        # Implement the function to add dynamic sample in prompt
        build the input_question and generated code the same format as existing opensearch index
        """
        input_list = input_question.split(", ")
        results = self.opensearch_vector_search.similarity_search(input_question, k=self.kexamples)  # our search query  # return 3 most relevant docs
        
        matched_example = ''
        for i in range(self.kexamples):
            result_json =  dumpd(results[i])
            example_str = str(result_json["kwargs"]["page_content"])
            matched_example+=example_str
        
        refined_prompt = f"""
    You are an experienced data analyst specialized in using Python to analyze time series data collected from vibration and temperature sensors. 
    You have a CSV containing sensor measurement data, with columns as:
    <csv metadata>
    timestamp
    sitename
    assetname
    sensorname
    temperature
    acceleration
    velocity
    temperatureML
    vibrationML
    </csv metadata>
    Here are a few examples to use Python code to analyze a similar csv file:
    <examples>
    {matched_example}
    </examples>

    Given the above csv file and past examples, please follow these steps to generate the python code output, think step by step:
    """+ "\n"
        for i in range(len(input_list)):
            refined_prompt += f"""
    <step {i+1}>
    {input_list[i]}
    </step {i+1}>
            """
        return refined_prompt

    def addtext_opensearch(self,input_question, generated_chat_code):
        """
        # Add new query to opensearch index to increase number of examples
        """
        reconstructed_json = {}
        reconstructed_json["question"]=input_question
        reconstructed_json["python_code"]=str(generated_chat_code)
        reconstructed_json["column_info"]="column_info: |\ntimestamp\nsitename\nassetname\nsensorname\ntemperature\nacceleration\nvelocity\ntemperatureML\nvibrationML\n"
    
        json_str = ''
        for key,value in reconstructed_json.items():
            json_str += key + ':' + value
        reconstructed_raw_text =[]
        reconstructed_raw_text.append(json_str)
        
        results = self.opensearch_vector_search.similarity_search_with_score(str(reconstructed_raw_text[0]), k=self.kexamples)  # our search query  # return 3 most relevant docs
        if (dumpd(results[0][1])<0.03):    ###No similar embedding exist, then add text to embedding
            response = self.opensearch_vector_search.add_texts(texts=reconstructed_raw_text, engine="faiss", index_name=self.indexname)
        else:
            response = "A similar embedding is already exist, no action."
        print(response)
        return response













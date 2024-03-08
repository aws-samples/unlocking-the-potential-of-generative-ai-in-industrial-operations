# Author: Julia Hu
#########################import library#############################
import json
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

############################create a class that can capture and display streamed output######################
class StreamProcessor(): 
    """"
    A class that can capture and display streamed output
    """
    def __init__(self, output_container):
        self.output_container = output_container
        self.combined_text = ""
            
    def streaming_handler(self, text_chunk):
        """
        append new text chunk and render combined tet
        """
        self.combined_text += text_chunk 
        self.combined_text = self.combined_text.replace("$","\$") #replace $ with \$
        self.output_container.write(self.combined_text) #render the combined text

class ImageProcessClass:
    """"
    A class for image chat 
    """
    def __init__(self,rekognition_client,bedrock_client):
        self.rekognition_client = rekognition_client
        self.bedrock_client = bedrock_client 

    def call_bedrock_claude(self, prompt_text, streaming_callback, max_tokens_to_sample=1024, temperature=1, top_k=250, top_p=1):
        """"
        A function to call claude model
        """
        model_id = "anthropic.claude-v2"
        body = {
            "prompt": "\n\nHuman:"+prompt_text+"\n\nAssistant:",
            "max_tokens_to_sample": max_tokens_to_sample
        }
        body_string = json.dumps(body)
        body = bytes(body_string, 'utf-8')
        response = self.bedrock_client.invoke_model_with_response_stream(
            modelId = model_id,
            contentType = "application/json",
            accept="*/*",
            body = body)
        stream = response.get('body')
        if stream:
            for event in stream: #process each event returned by the stream
                chunk = event.get('chunk')
                if chunk:
                    chunk_json = json.loads(chunk.get('bytes').decode())
                    streaming_callback(chunk_json["completion"]) #pass the latest chunk's text to the callback method
        
    def upload_image_detect_labels(self, bytes_data, language):
        """"
        A function to call Rekognition to detect labels and return a query string for the image. 
        """
        label_text = ''
        response = self.rekognition_client.detect_labels(
            Image={'Bytes': bytes_data},
            Features=['GENERAL_LABELS']
        )
        text_res = self.rekognition_client.detect_text(
            Image={'Bytes': bytes_data}
        )

        celeb_res = self.rekognition_client.recognize_celebrities(
            Image={'Bytes': bytes_data}
        )

        for celeb in celeb_res['CelebrityFaces']:
            label_text += celeb['Name'] + ' ' 

        for text in text_res['TextDetections']:
            label_text += text['DetectedText'] + ' '

        for label in response['Labels']:
            label_text += label['Name'] + ' '

        query = 'Identify and explain the equipment used in this image in ' +language+ ' in 200 words from these labels: '+ label_text +"\n Ensure that your answer is accurate and doesn’t contain any information not directly supported by the image and the label_text."
        
        return query
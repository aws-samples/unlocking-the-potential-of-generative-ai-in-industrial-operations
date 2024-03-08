# Author: Julia Hu, Sudeesh Sasidharan
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

class KnowledgebaseRAGClass:
    """"
    A class for knowledgebase RAG
    """
    def __init__(self, embeddings, aoss_client, index,k):
        self.embeddings = embeddings
        self.aoss_client = aoss_client
        self.index = index
        self.k = k

    def query_docs(self, query: str):
        """
        Convert the query into embedding and then find similar documents from AOSS
        Create a context out of the similar docs retrieved from the vector database
        by concatenating the text from the similar documents.
        """
        # embedding
        query_embedding = self.embeddings.embed_query(query)

        # query to lookup OpenSearch kNN vector. Can add any metadata fields based filtering
        # here as part of this query.
        query_qna = {
            "size": self.k,
            "query": {
                "knn": {
                "bedrock-knowledge-base-default-vector": {
                    "vector": query_embedding,
                    "k": self.k
                    }
                }
            }
        }

        # OpenSearch API call
        relevant_documents = self.aoss_client.search(
            body = query_qna,
            index = self.index
        )
        context = ""
        for r in relevant_documents['hits']['hits']:
            s = r['_source']
            
            context += f"{s['AMAZON_BEDROCK_TEXT_CHUNK']}\n"
            
        return relevant_documents, context
        
    def get_prompt(self, context, input_text):
        """
        Use prompt template to reformt prompt template
        """
        PROMPT_TEMPLATE = """Human: Answer the question based only on the information provided in few sentences.
        <context>
        {}
        </context>
        Include your answer in the <answer></answer> tags. Do not include any preamble in your answer.
        <question>
        {}
        </question>
        Assistant:"""
        prompt = PROMPT_TEMPLATE.format(context, input_text)
        return prompt

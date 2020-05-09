# pre-req setup start
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

# Download model
# download_model(model='bert-squad_1.1', dir='./models')

# Download pdf files from BNP Paribas public news
def download_pdf():
    import os
    import wget
    directory = './data/pdf/'
    models_url = [
      'https://invest.bnpparibas.com/documents/1q19-pr-12648',
      'https://invest.bnpparibas.com/documents/4q18-pr-18000',
      'https://invest.bnpparibas.com/documents/4q17-pr'
    ]

    print('\nDownloading PDF files...')

    if not os.path.exists(directory):
        os.makedirs(directory)
    for url in models_url:
        wget.download(url=url, out=directory)


# download_pdf()

# Convert the pdf files into a dataframe
df = pdf_converter(directory_path='./data/pdf/')
print(df.head())
print("pdf files converted")

# Instantiate cdQA pipeline from model
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)
# cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')

# Fit Retriever to documents
cdqa_pipeline.fit_retriever(df=df)


# pre-req setup end

# Evaluating Models
from cdqa.utils.converters import df2squad

# 1. convert pandas df into json file with SQuAD format
json_data = df2squad(df=df, squad_version='v1.1', output_dir='.', filename='dataset-name')

# 2. use annotator to add ground truth

# 3. evaluate the pipeline

from cdqa.utils.evaluation import evaluate_pipeline
# evaluate_pipeline(cdqa_pipeline, 'path-to-annotated-dataset.json')
# evaluate_pipeline(cdqa_pipeline, './data/SQuAD_1.1/dev-v1.1-short.json')

# 4. evaluate the reader
from cdqa.utils.evaluation import evaluate_reader

evaluation = evaluate_reader(cdqa_pipeline, './data/SQuAD_1.1/dev-v1.1-short.json')
print(evaluation)

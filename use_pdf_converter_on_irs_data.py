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

# execute a query
# query = 'How many contracts did BNP Paribas Cardif sell in 2019?'
questions = [
    'Who do custom animal farmers need to consult with before buying fertilizer?',
    'do I qualify for an automatic extension of time to file without filing Form 4868?',
    'Did the coronavirus pandemic extend the deadline to pay taxes?',
    'What is the new tax deadline?',
    'What is the tax payer advocate service?',
    'What is the job of the taxpayer advocate service?',
    'What does the TAS work to resolve?',
    'What is the taxpayer advocate number?',
    'How do I call the Taxpayer Advocate Service?',
    'How do I extend my tax return?',
    'when will I get my economic impact payment?',
    'What are the coronavirus scams for economic impact payments?',
    'Why is the IRS is asking me for financial information so that I can get economic impact relief?',
    "Why is the IRS calling  me asking to verify financial information?"
]
for question in questions:
    prediction = cdqa_pipeline.predict(question)
    print()
    print('------------------------------------------------------------------------')
    print('question: {}'.format(question))
    print('answer: {}\n'.format(prediction[0]))
    print('title: {}\n'.format(prediction[1]))
    print('paragraph: {}\n'.format(prediction[2]))

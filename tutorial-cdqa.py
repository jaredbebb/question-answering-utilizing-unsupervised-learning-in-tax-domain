import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

# Download data and models
download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
# download_model(model='bert-squad_1.1', dir='./models')

# Loading data and filtering / preprocessing the documents
# df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = pd.read_csv('data/bnpp_newsroom_v1.1/custom_tax_jlb.csv', converters={'paragraphs': literal_eval})
df = filter_paragraphs(df)

# Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1
# cdqa_pipeline = QAPipeline(reader='models/bert_qa_vCPU-sklearn.joblib')
cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')

# Fitting the retriever to the list of documents in the dataframe
# cdqa_pipeline.fit_retriever(X=df)
cdqa_pipeline.fit_retriever(df=df)

# Sending a question to the pipeline and getting prediction
# query = 'Since when does the Excellence Program of BNP Paribas exist?'
# query = 'Who should investors  consult with prior to investing?'
# query = 'Who do custom animal farmers need to consult with before buying fertilizer?'
queries = [
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
# prediction = cdqa_pipeline.predict(X=query)
for query in queries:
    prediction = cdqa_pipeline.predict(query)
    print()
    print('------------------------------------------------------------------------')
    print('query: {}'.format(query))
    print('answer: {}\n'.format(prediction[0]))
    print('title: {}\n'.format(prediction[1]))
    print('paragraph: {}\n'.format(prediction[2]))

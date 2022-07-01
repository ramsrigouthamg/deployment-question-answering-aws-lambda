import torch
import json
import numpy as np
from transformers import pipeline

nlp = pipeline('question-answering',model="/opt/ml/model", tokenizer='/opt/ml/model')

def lambda_handler(event, context):
    raw_string = r'{}'.format(event['body'])
    body = json.loads(raw_string)
    originaltext = body['text']


    res = nlp(originaltext)

    final = {'output':res}

    return {
        "statusCode": 200,
        "body": json.dumps(final)
    }

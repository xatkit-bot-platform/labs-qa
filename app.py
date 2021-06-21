from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertConfig, LongformerConfig

import torch
import logging

logging.basicConfig(level=logging.INFO)

MAX_TOKENS = 512
MAX_TOKENS_XL = 4096

modelNames = []
models = []
tokenizers = []

app = Flask(__name__)

def splitParagraph(tokenizer, question, paragraph):
    inputs = tokenizer(question, paragraph, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    if (len(input_ids) > MAX_TOKENS):
        p1 = paragraph[0:int(len(paragraph)/2)]
        p2 = paragraph[int(len(paragraph)/2):]
        return splitParagraph(tokenizer, question, p1) + splitParagraph(tokenizer, question, p2)
    else:
        return [paragraph]
    
    
@app.route('/split-qa', methods=['POST'])
def transformerSplitQA():    
    body = request.get_json()
    question = body["question"]
    corpus_raw = body["corpus"]
    corpus = []
    response = {}
    for i, modelName in enumerate(modelNames):
        tokenizer = tokenizers[i]
        model = models[i]
        
        inputs = tokenizer(question, corpus_raw, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]        
        if (len(input_ids) > MAX_TOKENS):
            corpus = list(filter(lambda x: x != "", corpus_raw.split('\n')))
            for i, paragraph in enumerate(corpus):
                paragraph_inputs = tokenizer(question, paragraph, add_special_tokens=True, return_tensors="pt")
                paragraph_input_ids = paragraph_inputs["input_ids"].tolist()[0]
                if (len(paragraph_input_ids) > MAX_TOKENS):
                    corpus[i:i+1] = splitParagraph(tokenizer, question, paragraph)
        else:
            corpus = [corpus_raw] 
            
        best_answer = ""
        context = ""
        best_score = 0 # Are best scores always > 0 ???
        paragraph_id = -1
        for i, paragraph in enumerate(corpus):
            inputs = tokenizer(question, paragraph, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            corpus_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            score_start = torch.max(answer_start_scores)
            score_end = torch.max(answer_end_scores)
            score = score_start + score_end
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            if (score > best_score and (not (answer in ["[CLS]", "[SEP]", "<s>", "", " "]))):
                best_score = score
                best_answer = answer
                context = paragraph
                paragraph_id = i

        response[modelName] = {
            "question": question,
            "answer": best_answer,
            "context": context,
            "paragraph_id": paragraph_id
        }
    return response, 200

@app.route('/set-models', methods=['POST'])
def setModels():
    body = request.get_json()
    names = body["modelNames"]
    for name in names:
        modelNames.append(name)
        logging.info("Loading " + name + " tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(name)
        logging.info(name + " tokenizer loaded")
        logging.info("Loading " + name + " model")
        model = AutoModelForQuestionAnswering.from_pretrained(name)
        logging.info(name + " model loaded")
        print("------------------------------------")
        models.append(model)
        tokenizers.append(tokenizer)
    response = {
        "status": "done"
        }
    return response, 200
    
@app.route('/transformer-qa', methods=['POST'])
def transformerQA():    
    body = request.get_json()
    question = body["question"]
    corpus = body["corpus"]
    response = {}
    for i, modelName in enumerate(modelNames):
        tokenizer = tokenizers[i]
        model = models[i]
        encoding = tokenizer(question, corpus, add_special_tokens=True, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].tolist()[0]
        corpus_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        outputs = model(**encoding)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        response[modelName] = {
            "question": question,
            "answer": answer,
            "beginPosition": answer_start.item(),
            "endPosition": answer_end.item()
        }
    return response, 200

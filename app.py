from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import logging

logging.basicConfig(level=logging.INFO)

name1 = "bert-large-uncased-whole-word-masking-finetuned-squad"
name2 = "deepset/roberta-base-squad2"
name = name1

logging.info("Loading " + name + " tokenizer")
tokenizer = AutoTokenizer.from_pretrained(name)
logging.info(name + " tokenizer loaded")

logging.info("Loading " + name + " model")
model = AutoModelForQuestionAnswering.from_pretrained(name)
logging.info(name + " model loaded")



app = Flask(__name__)

@app.route('/qa', methods=['POST'])
def processQA():    
    body = request.get_json()
    question = body["question"]
    corpus = body["corpus"]
    inputs = tokenizer(question, corpus, add_special_tokens=True, return_tensors="pt")
    print(inputs)
    print("========================================")
    input_ids = inputs["input_ids"].tolist()[0]
    corpus_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(corpus_tokens)
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    response = {
    	"answer": answer,
    	"beginPosition": str(answer_start.item()),
    	"endPosition": str(answer_end.item())
    }
    
    return response, 200
    
    
    
    
    

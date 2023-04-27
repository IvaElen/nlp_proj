import streamlit as st
import torch
import numpy as np
import transformers 
import pickle

def load_model():
    model_finetuned = transformers.AutoModel.from_pretrained(
        "nghuyong/ernie-2.0-base-en",
        output_attentions = False,
        output_hidden_states = False
    )
    model_finetuned.load_state_dict(torch.load('ErnieModel_imdb.pt'))
    tokenizer = transformers.AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
    return model_finetuned, tokenizer

def preprocess_text(text_input, max_len, tokenizer):
    input_tokens = tokenizer(
        text_input, 
        return_tensors='pt', 
        padding=True, 
        max_length=max_len,
        truncation = True
        )
    return input_tokens

def predict_sentiment(model, input_tokens):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    output = model(**input_tokens).pooler_output.detach().numpy()
    with open('LogReg_imdb_Ernie.pkl', 'rb') as file:
        cls = pickle.load(file)
    result = id2label[cls.predict(output)]
    return result

st.title('Text sentiment analysis by ErnieModel')

max_len = st.slider('Maximum word length', 0, 500, 250)

text_input = st.text_input("Enter some text about movie")
model, tokenizer = load_model()

if text_input:
    input_tokens = preprocess_text(text_input, max_len, tokenizer)
    output = predict_sentiment(model, input_tokens)
    st.write(output)
    
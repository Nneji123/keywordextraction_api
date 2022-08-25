from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from keybert import KeyBERT

class Doc(BaseModel):
    title: str
    text: str
    length: int
    count: int

app = FastAPI()

kw_model = KeyBERT(model='all-mpnet-base-v2')

@app.post("/keywords")
def update_item(doc: Doc):
    full_text = f"{doc.title}, {doc.text}"
    # kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(full_text, 
                                     keyphrase_ngram_range=(1, doc.length), 
                                     stop_words='english', 
                                     top_n=doc.count)
    keywords_list= list(dict(keywords).keys())
    return {"keywords": keywords_list}



import time

from google.cloud import storage
from bsbi import BSBIIndex
from compression import VBEPostings
from letor import Letor
import numpy as np
import os

client_storage = storage.Client()
bucket = client_storage.bucket("diagnosee-collections")
ranker = Letor()

def search(request):
    query = request.args.get("query")
    if query is None:
        return "No q", 400
    
    index = BSBIIndex(output_dir='index', postings_encoding=VBEPostings)
    docs = []
    
    start = time.time()
    retrieve = index.retrieve_tfidf(query, k=100)
    end = time.time()
    
    serp = {}

    for (score, doc) in retrieve:
        did = os.path.splitext(os.path.basename(doc))[0]
        did = int(did.split("\\")[1])
        with bucket.blob("collections/"+doc[2:].replace("\\","/")).open("r") as f:
            content = f.read()
            docs.append((did, content))
            serp[did] = content

    # simpan content dari docs sebagai unseen document untuk testing
    X_unseen = []
    for doc_id, doc in docs:
        X_unseen.append(ranker.features(query.split(), doc.split()))
    X_unseen = np.array(X_unseen)

    hasil = {}
    # melakukan prediksi & re-ranking dari 100 dokumen sebelumnya
    if len(X_unseen) != 0:
        scores = ranker.ranker.predict(X_unseen)
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
    
        for item in sorted_did_scores:
            hasil[item[0]] = serp[item[0]]

    duration = end - start      
    return {"duration": duration, "length": len(serp), "serp": hasil}, 200

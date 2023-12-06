# referensi: Tutorial Learning-to-Rank dengan LambdaMART

from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import lightgbm as lgb
import numpy as np
import random
from google.cloud import storage

client_storage = storage.Client()
bucket = client_storage.bucket("diagnosee-collections")

# process dataset yang digunakan untuk training
def get_dataset():
    documents = {}
    with bucket.blob("qrels-folder/train_docs.txt").open("r") as file:
        for line in file:
            idx = line.find(" ")
            doc_id = line[:idx]
            content = line[idx+1:]
            documents[doc_id] = content.split()
    return documents

# class untuk mendapatkan query dan qrels dari masing-masing tipe data
class Data:
    
    # inisialisasi dan pemanggilan function
    def __init__(self, documents, type):
        # type = train / val
        self.NUM_NEGATIVES = 1
        self.documents = documents
        self.type = type
        self.queries = {}
        self.q_docs_rel = {}
        self.group_qid_count = []
        self.dataset = []
        self.get_queries()
        self.get_qrels()
        self.group_count()

    # mendapatkan query
    def get_queries(self):
        with bucket.blob(f"qrels-folder/{self.type}_queries.txt").open("r") as file:
            for line in file:
                idx = line.find(" ")
                q_id = line[:idx]
                content = line[idx+1:]
                self.queries[q_id] = content.split()
    
    # mendapatkan qrel yang dikelompokkan berdasarkan q_id    
    def get_qrels(self):
        with bucket.blob(f"qrels-folder/{self.type}_qrels.txt").open("r") as file:
            for line in file:
                q_id, doc_id, rel = line.split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))
            
    # menghitung grup q_id yang ada
    def group_count(self):
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            # menambahkan satu negative (random sampling dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

# main class
class Letor():

    # inisialisasi, pembuatan LSI model dan LightGBM LambdaMART
    def __init__(self):
        self.NUM_LATENT_TOPICS = 200
        self.documents = get_dataset()
        self.dictionary = Dictionary()
        self.train_data = Data(self.documents, "train")
        self.val_data = Data(self.documents, "val")
        self.build_lsi()
        self.ranker()
        
    def build_lsi(self):
        # memasukkan konversi content dari docs ke Bag of Words
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        
        # membuat model LSI dengan 200 topik laten
        self.model = LsiModel(bow_corpus, num_topics = self.NUM_LATENT_TOPICS)
        return self.model

    #  menampilkan representasi vektor dari suatu doc maupun query
    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc):
        # mengambil representasi vektor dari query dan doc
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        
        # menghitung cosine distance antara query dan doc
        cosine_dist = cosine(v_q, v_d)
        
        # menghitung koefisien jaccard similarity antara query dan doc
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def ranker(self):
        # memisahkan feature dengan target
        X = []
        Y = []
        
        # melatih model menggunakan data dari train set
        # karena hasilnya lebih baik daripada validation set
        for (query, doc, rel) in self.train_data.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)


        X = np.array(X)
        Y = np.array(Y)
        
        # membuat model LightGBM LambdaMART
        self.ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)

        # melatih model menggunakan data dari train set
        # karena hasilnya lebih baik daripada validation set
        self.ranker.fit(X, Y,
                group = self.train_data.group_qid_count)
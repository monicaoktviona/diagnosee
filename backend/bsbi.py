# referensi: https://www.geeksforgeeks.org/zip-in-python/ ,
# slide perkuliahan untuk perhitungan score

import os
import pickle
import contextlib
import heapq
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm
import re

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.doc_length = {}
        self.avg_doc_length = 0

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir)) as merged_index:
            self.doc_length = merged_index.doc_length
            self.avg_doc_length = sum(self.doc_length.values()) / len(self.doc_length)
        

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmer = MPStemmer()
        stemmed = stemmer.stem_kalimat(content.lower())
        remover = StopWordRemoverFactory().create_stop_word_remover()
        stemmed = remover.remove(stemmed)
        tokenizer_pattern: str = r'\w+'
        stemmed = re.findall(tokenizer_pattern, stemmed)
        return stemmed

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        
        # inisialisasi
        td_pairs = []
        
        # loop setiap text file
        for file_name in next(os.walk(os.path.join(self.data_dir, block_path)))[2]:
            doc_id = self.doc_id_map[f'./{os.path.join(block_path, file_name)}']
            
            # membaca text file
            with open(os.path.join(self.data_dir, block_path, file_name), encoding="utf-8") as content:
                # melakukan preprocessing isi text file
                clean_words = self.pre_processing_text(content.read())
                # menyimpan hasil preprocessing dan mappingnya ke list
                for word in clean_words:
                    term_id = self.term_id_map[word]
                    td_pairs.append((term_id, doc_id))
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        
        # inisialisasi
        term_dict = {}
        # akses setiap term
        for term_id, doc_id in td_pairs:
            # cek apakah ada di dict, tambahkan jika tidak ada
            term_dict.setdefault(term_id, {})
            term_dict[term_id].setdefault(doc_id, 0)
            term_dict[term_id][doc_id] += 1
            
        # sort, mapping, dan tambahkan ke index
        for term_id in sorted(term_dict.keys()):
            doc_pairs = sorted(term_dict[term_id].items())
            unzip = list(zip(*doc_pairs))
            index.append(term_id, list(unzip[0]), list(unzip[1]))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve(self, query):
        # melakukan preprocessing untuk query
        clean_query = self.pre_processing_text(query)

        # inisialisasi
        query_postings = []
        
        # membaca file index
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir)) as merged_index:

            # mendapatkan postings list untuk setiap term di query
            for word in clean_query:
                # jika term tidak ditemukan, lanjut ke iterasi selanjutnya 
                if word not in self.term_id_map:
                    continue
                query_postings.append(merged_index.get_postings_list(self.term_id_map[word]))

        query_postings.sort(key=len)
        
        return query_postings

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        
        # mendapatkan posting list untuk masing-masing term
        query_postings = self.retrieve(query=query)

        n = len(self.doc_length)

        result = []
        # akses setiap term untuk dihitung scorenya
        for item in query_postings:
            df = len(item[0])
            idf = math.log10(n / df)
            tuples = []    
            for j in range(df):
                tf = 0
                if (item[1][j] > 0):
                    tf = (1 + math.log10(item[1][j]))
                score = tf * idf
                # simpan doc id dan score dalam tuple bentuk di list
                tuples.append((item[0][j], score))
            # merge hasilnya
            result = merge_and_sort_posts_and_tfs(result, tuples)
        
        # sort berdasarkan score
        result = sorted(result, key=lambda x: x[1], reverse=True)
                
        # mengambil top k tuples
        docs_count = len(result)
        if k < len(result):
            docs_count = k
        top_k = []
        for i in range(docs_count):
            top_k.append((result[i][1], self.doc_id_map[result[i][0]]))  
        return top_k
    
    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        
        # mendapatkan posting list untuk masing-masing term
        queries = self.retrieve(query=query)
        
        n = len(self.doc_length)
        result = []
            
        # akses setiap term untuk dihitung scorenya
        for item in queries:
            df = len(item[0])
            idf = math.log10(n / df)
            tuples = []
            for j in range(df):
                tf = item[1][j]
                numer = (k1 + 1) * tf
                denom = (k1 * ((1 - b) + b * self.doc_length[item[0][j]] / self.avg_doc_length)) + tf
                score = (numer / denom) * idf

                # simpan doc id dan score dalam tuple bentuk di list
                tuples.append((item[0][j], score))
                
            # merge hasilnya
            result = merge_and_sort_posts_and_tfs(result, tuples)
    
        # sort berdasarkan score
        result = sorted(result, key=lambda x: x[1], reverse=True)
        
        # mengambil top k tuples
        docs_count = len(result)
        if k < len(result):
            docs_count = k
        top_k = []
        for i in range(docs_count):
            top_k.append((result[i][1], self.doc_id_map[result[i][0]]))  
        return top_k
        

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir)) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir)) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), self.output_dir)))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!

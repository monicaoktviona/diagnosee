# referensi: https://www.w3schools.com/python/ref_dictionary_setdefault.asp

import pickle
import os
from google.cloud import storage

client_storage = storage.Client()
bucket = client_storage.bucket("diagnosee-collections")

class InvertedIndex:
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file; dan juga menyediakan
    mekanisme untuk menulis Inverted Index ke file (storage) saat melakukan indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict adalah konsep "Dictionary" yang merupakan bagian dari
        Inverted Index. postings_dict ini diasumsikan dapat dimuat semuanya
        di memori.

        Seperti namanya, "Dictionary" diimplementasikan sebagai python's Dictionary
        yang memetakan term ID (integer) ke 4-tuple:
           1. start_position_in_index_file : (dalam satuan bytes) posisi dimana
              postings yang bersesuaian berada di file (storage). Kita bisa
              menggunakan operasi "seek" untuk mencapainya.
           2. number_of_postings_in_list : berapa banyak docID yang ada pada
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : panjang postings list dalam
              satuan byte.
           4. length_in_bytes_of_tf_list : panjang list of term frequencies dari
              postings list terkait dalam satuan byte

    terms: List[int]
        List of terms IDs, untuk mengingat urutan terms yang dimasukan ke
        dalam Inverted Index.

    """

    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Nama yang digunakan untuk menyimpan files yang berisi index
        postings_encoding : Lihat di compression.py, kandidatnya adalah StandardPostings,
                        GapBasedPostings, dsb.
        directory (str): directory dimana file index berada
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Untuk keep track urutan term yang dimasukkan ke index
        # key: doc ID (int), value: document length (number of tokens)
        self.doc_length = {}
        # Ini nantinya akan berguna untuk normalisasi Score terhadap panjang
        # dokumen saat menghitung score dengan TF-IDF atau BM25

    def __enter__(self):
        """
        Memuat semua metadata ketika memasuki context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator untuk List yang berisi urutan term yang masuk ke
                index saat konstruksi. ---> term_iter
            3. doc_length, sebuah python's dictionary yang berisi key = doc id, dan
                value berupa banyaknya token dalam dokumen tersebut (panjang dokumen).
                Berguna untuk normalisasi panjang saat menggunakan TF-IDF atau BM25
                scoring regime; berguna untuk untuk mengetahui nilai N saat hitung IDF,
                dimana N adalah banyaknya dokumen di koleksi

        Metadata disimpan ke file dengan bantuan library "pickle"

        Perlu memahani juga special method __enter__(..) pada Python dan juga
        konsep Context Manager di Python. Silakan pelajari link berikut:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Membuka index file
        self.index_file = bucket.blob(self.index_file_path).open('rb')

        # Kita muat postings dict dan terms iterator dari file metadata
        with bucket.blob(self.metadata_file_path).open('rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan postings_dict dan terms ketika keluar context"""
        # Menutup index file
        self.index_file.close()

        # Menyimpan metadata (postings dict dan terms) ke file metadata dengan bantuan pickle
        with bucket.blob(self.metadata_file_path).open('wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __iter__(self):
        return self

    def reset(self):
        """
        Kembalikan file pointer ke awal, dan kembalikan pointer iterator
        term ke awal
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()  # reset term iterator

    def __next__(self):
        """
        Class InvertedIndexReader juga bersifat iterable (mempunyai iterator).
        Silakan pelajari:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        Ketika instance dari kelas InvertedIndexReader ini digunakan
        sebagai iterator pada sebuah loop scheme, special method __next__(...)
        bertugas untuk mengembalikan pasangan (term, postings_list, tf_list) berikutnya
        pada inverted index.

        PERHATIAN! method ini harus mengembalikan sebagian kecil data dari
        file index yang besar. Mengapa hanya sebagian kecil? karena agar muat
        diproses di memori. JANGAN MEMUAT SEMUA INDEX DI MEMORI!
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[
            curr_term]
        postings_list = self.postings_encoding.decode(
            self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(
            self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Kembalikan sebuah postings list (list of docIDs) beserta list
        of term frequencies terkait untuk sebuah term (disimpan dalam
        bentuk tuple (postings_list, tf_list)).

        PERHATIAN! method tidak boleh iterasi di keseluruhan index
        dari awal hingga akhir. Method ini harus langsung loncat ke posisi
        byte tertentu pada file (index file) dimana postings list (dan juga
        list of TF) dari term disimpan.
        """
        # TODO
        
        # mengambil informasi term dari dict
        position, num, length_of_postings, length_of_tf = self.postings_dict[term]
        
        # mencari term di index
        self.index_file.seek(position)
        
        # mendapatkan postings list untuk term
        postings_list = self.index_file.read(length_of_postings)
        postings_list = self.postings_encoding.decode(postings_list)
        
        # mendapatkan tf list untuk term
        tf_list = self.index_file.read(length_of_tf)
        tf_list = self.postings_encoding.decode_tf(tf_list)
        
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya menulis secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __enter__(self):
        self.index_file = bucket.blob(self.index_file_path).open('wb')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Menambahkan (append) sebuah term, postings_list, dan juga TF list 
        yang terasosiasi ke posisi akhir index file.

        Method ini melakukan 4 hal:
        1. Encode postings_list menggunakan self.postings_encoding (method encode),
        2. Encode tf_list menggunakan self.postings_encoding (method encode_tf),
        3. Menyimpan metadata dalam bentuk self.terms, self.postings_dict, dan self.doc_length.
           Ingat kembali bahwa self.postings_dict memetakan sebuah termID ke
           sebuah 4-tuple: - start_position_in_index_file
                           - number_of_postings_in_list
                           - length_in_bytes_of_postings_list
                           - length_in_bytes_of_tf_list
        4. Menambahkan (append) bystream dari postings_list yang sudah di-encode dan
           tf_list yang sudah di-encode ke posisi akhir index file di harddisk.

        Jangan lupa update self.terms dan self.doc_length juga ya!

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
        - Anda mungkin mau membaca tentang Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
          Di link ini juga bisa kita pelajari bagaimana menambahkan informasi
          ke bagian akhir file.
        - Beberapa method dari object file yang mungkin berguna seperti seek(...)
          dan tell()

        Parameters
        ----------
        term:
            term atau termID yang merupakan unique identifier dari sebuah term
        postings_list: List[Int]
            List of docIDs dimana term muncul
        tf_list: List[Int]
            List of term frequencies
        """
        # TODO
        
        # encode postings list
        encoded_postings = self.postings_encoding.encode(postings_list)
        
        # encode tf list
        encoded_tf = self.postings_encoding.encode_tf(tf_list)
        
        # menambahkan term ke list dan dict
        self.terms.append(term)
        self.postings_dict[term] = (self.index_file.tell(), len(postings_list), len(encoded_postings), len(encoded_tf))
        for i in range(len(postings_list)):
            doc_id = postings_list[i]
            self.doc_length.setdefault(doc_id, 0)
            self.doc_length[doc_id] += tf_list[i]

        # menambahkan term ke index
        self.index_file.write(encoded_postings)
        self.index_file.write(encoded_tf)


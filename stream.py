# Import Library
import pickle
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Menyambungkan ke Google Spread Sheets
conn = st.connection('gsheets', type=GSheetsConnection)

# Menghubungkan ke data spread Sheet
existing_data = conn.read(worksheet="berita", usecols=list(range(3)), ttl=5)
existing_data = existing_data.dropna(how="all")

# Text Processing
# Membuat Fungsi CaseFolding
import re
def casefolding(text):
  text = text.lower()                                   # Mengubah huruf kecil
  text = re.sub(r'https?://\S+|www\.\s+', ' ', text)    # Menghapus hyperlinks
  text = re.sub(r',',' ',text)                          # Menghapus koma
  text = re.sub(r'[-+]?[0-9]+', ' ', text)              # Menghapus angka
  text = re.sub(r'[^\w\s]', ' ', text)                  # Menghapus semua karakter yg bukan huruf dan spasi
  text = text.strip()                                   # menghapus spasi berlebih dan karakter
  return text

# Membuat Fungsi Normalisasi Teks
key_norm = pd.read_csv('key_norm.csv')
def text_normalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
  if (key_norm['singkat'] == word).any()
  else word for word in text.split()
  ])

  text = str.lower(text)
  return text

# Membuat fungsi Stopword Removal
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

stopwords_indonesia = stopwords.words('indonesian')
more_stopword = []
stopwords_indonesia = stopwords_indonesia + more_stopword

def remove_stopword(text):
  clean_words = []
  words = word_tokenize(text)
  for word in words:
    if word not in stopwords_indonesia:
      clean_words.append(word)
  return " ".join(clean_words)

# Merubah Kata menjadi Kata dasar
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Membuat Fungsi untuk Stemming bahasa Indonesia
def stemming(text):
  text = stemmer.stem(text)
  return text

# Membuat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing(text):
  text = casefolding(text)
  text = text_normalize(text)
  text = remove_stopword(text)
  text = stemming(text)
  return text

#load save model
model_NB = pickle.load(open('model_NB.sav', 'rb'))

model_SVM = pickle.load(open('model_SVM.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary = set(pickle.load(open('new_selected_feature_tf-idf.sav', 'rb'))))

# bikin side bar

# Judul halaman
st.title('klasifikasi Berita Pemilu')

# Bikin menu sidebar
menu = st.sidebar.selectbox('Pilih Fungsi',('Klasifikasi', 'History'))

if menu == 'Klasifikasi' :
# Input Berita
  teks = st.text_input('Masukan Teks Berita')
  input = text_preprocessing(teks)

  detect_NB = ''
  detect_SVM = ''

  if st.button('Hasil Deteksi'):
    #Fungsi Prediksi SVM
    predict_NB = model_NB.predict(loaded_vec.fit_transform([input]))

    if (predict_NB == 0):
      detect_NB = 'Fake News'

    elif (predict_NB == 1):
      detect_NB = 'Real News'
    
    #Fungsi Prediksi SVM
    dense_data_input = loaded_vec.fit_transform([input]).toarray()
    predict_SVM = model_SVM.predict(dense_data_input)

    if (predict_SVM == 0):
      detect_SVM = 'Fake News'
    if (predict_SVM == 1):
      detect_SVM = 'Real News'

    # memasukan data ke dataBaru
    dataBaru = pd.DataFrame(
      [
        {
          "Berita" : teks,
          "HasilNB" : detect_NB,
          "HasilSVM" : detect_SVM
        }
      ]
    )

    # menggabungkan dataBaru ke dalam data sekarang
    updated_df = pd.concat([existing_data, dataBaru], ignore_index=True)

    # update data spread sheet dengan data sekarang
    conn.update(worksheet="berita", data=updated_df)
    
    #hasil
    st.success(f'Prediksi Naive Bayes = {detect_NB}')
    st.success(f'Prediksi Support Vector Machine = {detect_SVM}')
    st.success("data spreadsheet sudah di perbarui")
    
# menunjukan hasil
if menu == 'History':
  st.dataframe(existing_data)
  # delete
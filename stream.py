# Import Library
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import snowflake.snowpark as snowpark
from snowflake.snowpark.types import StructType, StructField, StringType

# Define a function to create a Snowpark session
connection_parameters = {
    'account': st.secrets["snowflake"]["account"],
    'user': st.secrets["snowflake"]["user"],
    'password': st.secrets["snowflake"]["password"],
    'role': st.secrets["snowflake"]["role"],
    'warehouse': st.secrets["snowflake"]["warehouse"],
    'database': st.secrets["snowflake"]["database"],
    'schema': st.secrets["snowflake"]["schema"]
}
session = snowpark.Session.builder.configs(connection_parameters).create()

# Define the schema for the dummy data
schema = StructType([
    StructField("BERITA", StringType()),
    StructField("HASILNB", StringType()),
    StructField("HASILSVM", StringType())
])
    
# Text Processing
# CaseFolding
import re
def casefolding(text):
  text = text.lower()                                   # Mengubah huruf kecil
  text = re.sub(r'https?://\S+|www\.\s+', ' ', text)    # Menghapus hyperlinks
  text = re.sub(r',',' ',text)                          # Menghapus koma
  text = re.sub(r'[-+]?[0-9]+', ' ', text)              # Menghapus angka
  text = re.sub(r'[^\w\s]', ' ', text)                  # Menghapus semua karakter yg bukan huruf dan spasi
  text = text.strip()                                   # menghapus spasi berlebih dan karakter
  return text

# Normalisasi Teks
key_norm = pd.read_csv('key_norm.csv')
def text_normalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
  if (key_norm['singkat'] == word).any()
  else word for word in text.split()
  ])

  text = str.lower(text)
  return text

# Stopword Removal
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

# Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
  text = stemmer.stem(text)
  return text

# text preprocessing pipeline
def text_preprocessing(text):
  text = casefolding(text)
  text = text_normalize(text)
  text = remove_stopword(text)
  text = stemming(text)
  return text

# load save model
model_NB = pickle.load(open('model_NB.sav', 'rb'))

model_SVM = pickle.load(open('model_SVM.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary = set(pickle.load(open('new_selected_feature_tf-idf.sav', 'rb'))))



# Judul halaman
st.title('CekBeritaPemilu.ID')

# Navbar
selected = option_menu(
  menu_title=None,
  options=["Beranda", "Klasifikasi", "Riwayat"],
  icons=['house', 'hr', 'clock-history'],
  orientation="horizontal"
)

# Beranda
if selected == "Beranda" :
# Tentang Kami
  st.subheader('Tentang Kami')
  st.markdown('CekBeritaPemilu.ID merupakan platform yang dirancang dan dikembangkan untuk mengklasifikasi berita mengenai pemilu yang tersebar di masyarakat dan dilabeli kedalam berita benar dan berita salah, dengan mengguanakan metode _Text Mining_ untuk mengolah data teks dan diklasifikasikan menggunakan metode _Naive Bayes_ dan _Support Vector Machine_, data teks akan diklasifikasikan secara otomatis dan akurat kedalam berita benar dan berita salah.')
# Metode yang Digunakan
  st.subheader('Metode yang Digunakan')
  st.markdown('_Text Mining_ merupakan proses pengolahan data berupa teks untuk mendapatkan informasi dari hasil pengolahan data teks tersebut, proses _Text Mining_ yang dilakukan pada program ini terdiri dari proses berikut :')
# Text Processing
  with st.expander("Text Processing"):
    st.write('''
             <ul>  
                <li>
                  <b>Case Folding</b>
                  <p>
                    Proses penyeragaman data teks untuk mengoptimalkan pengolahan data teks, proses Case Folding yang dilakukan diantaranya :
                    <ul>
                      <li>Mengubah huruf menjadi huruf kecil</li>
                      <li>Menghapus Hyperlink</li>
                      <li>Menghapus tanda koma</li>
                      <li>menghapus angka</li>
                      <li>Menghapus semua spasi dan karakter yang bukan huruf</li>
                    </ul>
                  </p>
                </li>
             
                <li>
                  <b>Normalisasi Teks</b>
                  <p>
                    Proses penyaringan kata-kata singkatan dan kurang jelas didalam dataset menjadi kata-kata yang lebih lengkap.
                  </p>
                </li>
             
                <li>
                  <b>Stopword Removal</b>
                  <p>
                    Proses penyaringan kata-kata sambung dan umum yang sering muncul dalam kalimat, tetapi tidak memberikan informasi yang penting mengenai data tersebut. proses ini menggunakan library stopwords indonesia.
                  </p>
                </li>
             
                <li>
                  <b>Stemming</b>
                  <p>
                    Proses pengubahan bentuk kata menjadi kata dasar tanpa imbuhan. proses ini menggunakan library sastrawi.
                  </p>
                </li>
             </ul>''', unsafe_allow_html=True)
    
# TfIdf vectorizer
  with st.expander("TFIDF Vectorizer"):
    st.write('''Proses Pembobotan kata dari kata yang sering muncul didalam dataset
             <ul>
                <li>
                  <b>TF(Term Frequency)</b>
                  <p>Proses ini menghitung seberapa sering suatu kata/token yang muncul di dalam dokumen.</p>
                </li>
                <li>
                  <b>IDF(Invers Document Frequency)</b>
                  <p>Proses ini menilai seberapa penting suatu kata/token.</p>
                </li>
             </ul>
             ''', unsafe_allow_html=True)

# Modeling
  with st.expander("Modeling"):
    st.write('''Proses untuk mengklasifikasi data yang telah dimasukan menggunakan metode klasifikasi yang digunakan, yaitu :
             <ul>
                <li>
                  <b>Naive Bayes</b>
                  <p>
                    Metode klasifikasi dengan menggunakan metode probabilitas dan statistik untuk memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya. Ciri utama dari Naïve Bayes Classifier ini adalah asumsi yg sangat kuat (naïf) akan independensi dari masing-masing kondisi/kejadian.
                  </p>
                </li>
             </ul>
             <ul>
                <li>
                  <b>Support Vector Machine</b>
                  <p>
                    Metode Klasifikasi yang bekerja atas prinsip Structural Risk Minimization(SRM) dengan tujuan menemukan hyperplane terbaik yang memisahkan dua buah class pada input space.
                  </p>
                </li>
             </ul>
             ''', unsafe_allow_html= True)
 

  
# klasifikasi
if selected == "Klasifikasi" :
  st.subheader("Cek Berita")
# Input Berita
  teks = st.text_input('Masukan Teks Berita')
  input = text_preprocessing(teks)

  detect_NB = ''
  detect_SVM = ''

  if st.button('Hasil Deteksi'):
    #Fungsi Prediksi SVM
    predict_NB = model_NB.predict(loaded_vec.fit_transform([input]))

    if (predict_NB == 'Negatif'):
      detect_NB = 'Fake News'

    elif (predict_NB == 'Positif'):
      detect_NB = 'Real News'
    
    #Fungsi Prediksi SVM
    dense_data_input = loaded_vec.fit_transform([input]).toarray()
    predict_SVM = model_SVM.predict(dense_data_input)

    if (predict_SVM == 'Negatif'):
      detect_SVM = 'Fake News'
    if (predict_SVM == 'Positif'):
      detect_SVM = 'Real News'

    # memasukan data ke dataBaru
    dataBaru = pd.DataFrame(
      [
        {
          "BERITA" : teks,
          "HASILNB" : detect_NB,
          "HASILSVM" : detect_SVM
        }
      ]
    )

    # Update the Snowflake table with the combined data
    session.write_pandas(dataBaru, "BERITA")

    #hasil
    st.success(f'Prediksi Naive Bayes = {detect_NB}')
    st.success(f'Prediksi Support Vector Machine = {detect_SVM}')
    
# menunjukan hasil
if selected == "Riwayat":
  # Reload the updated data to display
  updated_db = session.table("BERITA").to_pandas()

  st.subheader("Hasil Cek Berita")
  st.dataframe(updated_db)

# Close the session
session.close()
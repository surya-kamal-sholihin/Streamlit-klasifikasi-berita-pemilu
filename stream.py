# Import Library
import pickle
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Menyambungkan ke Google Spread Sheets
conn = st.connection('gsheets', type=GSheetsConnection)

# Menghubungkan ke data spread Sheet
existing_data = conn.read(worksheet="klasifikasi", usecols=list(range(3)), ttl=5)
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

st.title('CekBerita.ID')  
# Membuat Menu
selected = option_menu(
  menu_title=None,
  options=['Beranda','Cek Berita', 'Hasil Cek'],
  icons=['house','hr','clock-history'],
  default_index=0,
  orientation="horizontal")

# Halaman Home
if selected == 'Beranda' :  

# Tentang Website
  st.subheader('Tentang Kami')
  st.markdown(f"""\
            <div style='text-align: justify;'>
              CekBeritaPemilu.ID adalah platform yang dirancang dan di kembangkan untuk mengklasifikasi berita mengenai pemilu yang tersebar di masyarakat dan dilabeli kedalam berita benar dan berita salah, dengan Mengunakan metode Text Mining untuk mengolah data teks dan di klasifikasikan menggunakan teknik Naive Bayes dan teknik Support Vector Machine, data teks akan di klasifikasikan secara otomatis dan Akurat kedalam berita benar dan berita salah.
            </div> 
            <br>""",
            unsafe_allow_html=True)

# Metode yang digunakan
  st.subheader('Metode yang Digunakan')
  st.markdown(f"""\
            <div style='text-align: justify;'>
              Text Mining merupakan proses penggalian data berupa teks yang diolah untuk mendapatkan informasi dari hasil pengolahan data teks tersebut. Adapun proses Metode Teks Mining yang program kami lakukan adalah sebagai berikut :
            </div> 
            <br>""",
            unsafe_allow_html=True)
  
  # Text Preprocessing
  with st.expander("Text Processing") :
    st.markdown(f"""\
            <div style='text-align: justify;'>
              Merupakan proses penyiapkan teks untuk diolah dan dibersihkan agar mendapatkan hasil yang maksimal, tahapan teks Processing yang program kami lakukan :
            </div>
            <ul>
                <li><b>Case Folding</b> <br> <p>Proses mengubah data menjadi huruf kecil, lalu data dirapikan dengan menghapus karakter yang tidak berguna dalam pemrosesan data.</p></li>
                <li><b>Normalisasi Teks</b> <br>  <p>proses mengubah kata singkatan, asing atau yang belum jelas ke kata yang lebih jelas/baku.</p></li>
                <li><b>Filtering (StopWord Removal)</b> <br>  <p>Proses Memisahkan data teks dari kata-kata yang yang kurang berguna dan tidak berarti dalam prosesan data.</p></li>
                <li><b>Stemming</b> <br>  <p>Proses Mengubah kata dalam dataset menjadi bentuk dasar dari kata tersebut.</p></li>
            </ul>
            <br>""",
          unsafe_allow_html=True)
    
  # Tf-Idf Vectorizer
  with st.expander("Tf-Idf Vectorizer") :
    st.markdown(f"""\
            <div style='text-align: justify'>
              Proses pembobotan dengan munggunakan teknik TF-IDF untuk melihat seberapa penting suatu teks dalam kalimat tersebut.
            </div>
            <br>""",
        unsafe_allow_html=True)
    
  # Modeling
  with st.expander("Modeling") :
    st.markdown(f"""\
            <div style='text-align: justify'>
              Proses Membangun Model klasifikasi data yang telah diproses dan diolah sebelumnya menggunaakan teknik Naive Bayes dan Teknik Support Vector Machine, lalu menampilkan hasil dari klasifikasi data tersebut.
            </div>
            <br>
            """,
        unsafe_allow_html=True)
  
# Text Preprocessing
  st.subheader('Teknik yang Digunakan')
  st.markdown(f"""\
            <div style='text-align: justify'>
              Dalam proses klasifikasi kami menggunakan teknik kami menggunakan teknik yang berbeda untuk melihat perbandingan akurasi dari kedua teknik, berikut adalah teknik yang kami gunakan dalam program kami :
            </div>
            <br>""",
        unsafe_allow_html=True)

  # Naive bayes
  with st.expander("Naive Bayes") :
    st.markdown(f"""\
            <div style='text-align: justify'>
              Naïve Bayes Classifier merupakan sebuah teknik klasifikasi yang berakar pada teorema Bayes. Metode pengklasifikasian dengan menggunakan metode probabilitas dan statistik yang dikemukakan oleh ilmuwan Inggris Thomas Bayes, yaitu memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya sehingga dikenal sebagai Teorema Bayes. Ciri utama dr Naïve Bayes Classifier ini adalah asumsi yg sangat kuat (naïf) akan independensi dari masing-masing kondisi/kejadian.
            </div>
            <br>
            """,
        unsafe_allow_html=True)
    
  # Support Vector Machine
  with st.expander("Support Vector Machine") :
    st.markdown(f"""\
            <div style='text-align: justify'>
              Support Vector Machine(SVM) adalah metode machine learning yang bekerja atas prinsip Structural Risk Minimization (SRM) dengan tujuan menemukan hyperplane terbaik yang memisahkan dua buah class pada input space. Menurut Vapnik SVM merupakan suatu teknik untuk menemukan hyperplane yang bisa memisahkan dua set data dari dua kelas yang berbeda.
            </div>
            <br>
            """,
        unsafe_allow_html=True)

# Halaman Klasifikasi
if selected == 'Cek Berita' :
  # Title
  st.subheader('Cek Berita')

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
    
    #hasil
    st.success(f'Prediksi Naive Bayes = {detect_NB}')
    st.success(f'Prediksi Support Vector Machine = {detect_SVM}')

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
    conn.update(worksheet="klasifikasi", data=updated_df)
    
# Halaman Hasil
if selected == 'Hasil Cek':
  # title
  st.subheader('Hasil Cek Berita')
  # History
  st.dataframe(existing_data)
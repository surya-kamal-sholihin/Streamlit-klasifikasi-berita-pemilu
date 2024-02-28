# Import Library
import pickle
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Menyambungkan ke Google Spread Sheets
conn = st.connection('gsheets', type=GSheetsConnection)

# Menghubungkan ke data spread Sheet
existing_data = conn.read(worksheet="klasifikasi", usecols=list(range(3)), ttl=5)
existing_data = existing_data.dropna(how="all")


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

  detect_NB = ''
  detect_SVM = ''

  if st.button('Hasil Deteksi'):
    #Fungsi Prediksi SVM
    predict_NB = model_NB.predict(loaded_vec.fit_transform([teks]))

    if (predict_NB == 0):
      detect_NB = 'Fake News'

    elif (predict_NB == 1):
      detect_NB = 'Real News'
    
    #Fungsi Prediksi SVM
    dense_data_input = loaded_vec.fit_transform([teks]).toarray()
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
    conn.update(worksheet="klasifikasi", data=updated_df)
    
    #hasil
    st.success(f'Prediksi Naive Bayes = {detect_NB}')
    st.success(f'Prediksi Support Vector Machine = {detect_SVM}')
    st.success("data spreadsheet sudah di perbarui")
    
# menunjukan hasil
if menu == 'History':
  st.dataframe(existing_data)
  # delete
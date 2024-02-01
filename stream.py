# Import Library
!pip install scikit-learn
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
model_NB = pickle.load(open('model_NB.sav', 'rb'))

model_SVM = pickle.load(open('model_SVM.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary = set(pickle.load(open('new_selected_feature_tf-idf.sav', 'rb'))))

# Judul halaman
st.title('klasifikasi Berita Pemilu')

# Input Berita
clean_teks = st.text_input('Masukan Teks Berita')

fakenews_detection_NB = ''
fakenews_detection_SVM = ''

if st.button('Hasil Deteksi'):
  #Fungsi Prediksi SVM
  predict_NB = model_NB.predict(loaded_vec.fit_transform([clean_teks]))

  if (predict_NB == 0):
    fakenews_detection_NB = 'Fake News'

  elif (predict_NB == 1):
    fakenews_detection_NB = 'Real News'
  
  #Fungsi Prediksi SVM
  dense_data_input = loaded_vec.fit_transform([clean_teks]).toarray()
  predict_SVM = model_SVM.predict(dense_data_input)

  if (predict_SVM == 0):
    fakenews_detection_SVM = 'Fake News'
  if (predict_SVM == 1):
    fakenews_detection_SVM = 'Real News'

st.success(f'Prediksi Naive Bayes = {fakenews_detection_NB}')
st.success(f'Prediksi Support Vector Machine = {fakenews_detection_SVM}')

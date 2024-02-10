# Import Library
import pickle
import streamlit as st
import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Connect to mysql server
mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="mendokusai",
            database="streamlit",
        )
mycursor = mydb.cursor()
print('Connection Established')

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

    # memasukan ke dalam data base
    sql = "insert into hasil(berita, hasilNB, hasilSVM) values(%s,%s,%s)"
    val = (teks, detect_NB, detect_SVM)
    mycursor.execute(sql, val)
    mydb.commit()
    
    #hasil
    st.success(f'Prediksi Naive Bayes = {detect_NB}')
    st.success(f'Prediksi Support Vector Machine = {detect_SVM}')
    st.success('Record Created Successfully!!!')

# menunjukan hasil
if menu == 'History':
  st.subheader('Hasil Prediksi')
  mycursor.execute('select * from hasil')
  result = mycursor.fetchall()
  df = pd.DataFrame(result, columns=mycursor.column_names)
  st.dataframe(df)
  
  # delete
  id = st.number_input('masukan angka')
  if st.button('Delete'):
    sql = 'delete from hasil where id = %s'
    val = (id,)
    mycursor.execute(sql, val)
    mydb.commit()
    st.success('Record Deleted Successfully!!!')
import streamlit as st
import pickle
import os
from sklearn.datasets import load_iris

#load dataset
iris = load_iris()

model_directory = r"model"

# Gunakan os.path.join() untuk menggabungkan direktori dan file model pickle
model_path = os.path.join(model_directory, 'knn_dt_iris_model.pkl')

# Periksa apakah file ada di direktori yang ditentukan
if os.path.exists(model_path):
    try:
        # Muat model dari file pickle
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        knn_model = loaded_model[0]
        dt_model = loaded_model[1]

        #bagian Streamlit App
        st.title("Prediksi Kelas Bunga Iris")

        st.write("""
        Aplikasi ini dapat memprediksi spesies bunga Iris berdasarkan input panjang dan lebar sepal serta petal.
        """)

        #buat input fitur dari pengguna
        sepal_length = st.number_input("Panjang Sepal (cm)",min_value=4.3, max_value=7.9, step=0.1)
        sepal_width = st.number_input("Lebar Sepal (cm):", min_value=2.0, max_value=4.4, step=0.1)
        petal_length = st.number_input("Panjang Petal (cm):", min_value=1.0, max_value=6.9, step=0.1)
        petal_width = st.number_input("Lebar Petal (cm):", min_value=0.1, max_value=2.5, step=0.1)

        #prediksi kelas iris berdasarkan input


        if st.button("Prediksi Kelas!"):
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

            knn_model_prediction = knn_model.predict(input_data)
            dt_model_prediction = dt_model.predict(input_data)

            knn_predict_class = iris.target_names[knn_model_prediction][0]
            dt_predict_class = iris.target_names[dt_model_prediction][0]


        #tampilkan hasil prediksi
        st.write(f"Kelas Iris yang diprediksi oleh KNN: **{knn_predict_class}**")
        st.write(f"Kelas Iris yang diprediksi Decision Tree: **{dt_predict_class}**")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
else:
    print("File 'knn_dt_iris_model.pkl' tidak ditemukan di direktori yang ditentukan.")

    
import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    

    # Dataframe selection
    st.markdown("<h1 align='center'> <b> Sistem blalalala Prediksi Cryptocurrency Ekosistem Metaverse", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Sebuah aplikasi prediksi yang intuitif dan kuat yang dirancang untuk menyederhanakan proses membangun dan mengevaluasi model pembelajaran mesin. Sistem Prediksi ini menggunakan Algoritma Long Short Term Memory dan Gated Recurrent Unit, sistem ini diharapkan dapat menjadi landasan untuk strategi perdagangan yang lebih cerdas dan keputusan investasi yang lebih baik dalam cryptocurrency.", unsafe_allow_html=True)
    
    st.divider()
    
    #overview
    new_line()
    st.markdown("<h2 style='text-align: center; '>🗺️ Gambaran Umum</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Ketika membangun model Prediksi ,ada serangkaian langkah untuk menyiapkan data dan membangun model. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    - **📦 Pengumpulan Data**: proses pengumpulan data dari berbagai sumber seperti pustaka yfinance, file CSV, database, API, dll.<br> <br>
    - **🧹 Data Cleaning**: proses pembersihan data dengan menghapus duplikasi, menangani nilai yang hilang dll. Langkah ini sangat penting karena seringkali data tidak bersih dan mengandung banyak nilai yang hilang dan outlier. <br> <br>
    - **⚙️ Data Preprocessing**: proses mengubah data ke dalam format yang sesuai untuk analisis. Hal ini termasuk menangani fitur kategorikal, fitur numerik, penskalaan dan transformasi, dll.. <br> <br>
    - **💡 Feature Engineering**: proses yang memanipulasi fitur itu sendiri. Terdiri dari beberapa langkah seperti ekstraksi fitur, transformasi fitur, dan pemilihan fitur. <br> <br>
    - **✂️ Splitting the Data**: proses membagi data menjadi set pelatihan, validasi, dan pengujian. Set pelatihan digunakan untuk melatih model, set validasi digunakan untuk menyetel hiperparameter, dan set pengujian digunakan untuk mengevaluasi model.. <br> <br>
    - **🧠 Building Machine Learning Models**: Model yang digunakan pada aplikasi ini adalah Long Short Term Memory dan Gated Recurrent Unit. Dalam konteks deep learning, LSTM dan GRU adalah salah satu arsitektur dari RNN yang sering digunakan <br> <br>
    - **⚖️ Evaluating Machine Learning Models**: proses mengevaluasi model prediksi dengan menggunakan metrik seperti Mean Absolute Percentage Error  (MAPE), Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE). <br> <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Pada bagian membangun model user memasukkan nilai masing masing hyperparameter.Hiperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model:
    
    - **⏱ Time Seps**: adalah parameter berupa nilai integer yang menentukan berapa banyak dataset latih yang digunakan untuk memprediksi harga kripto kedepannya. <br> <br>
    - **🧾 Units**: adalah sebuah nilai integer yang menentukan berapa banyak layer atau lapisan LSTM dan GRU yang dibangun dan digunakan didalam jaringan saraf. <br> <br>
    - **💣 Dropout**: Adalah sebuah teknik untuk menonaktifkan beberapa fungsi pada cell untuk mencegah terjadinya overfitting. <br> <br>
    - **📚 Learning Rate**: proses mengatur seberapa besar langkah perubahan bobot yang dilakukan selama proses pembelajaran. <br> <br>
    - **📠 Epochs**: adalah hyperparameter yang menggunakan nilai integer yang menentukan jumlah berapa kali program akan bekerja mengolah seluruh dataset latih <br> <br>
    - **🧠 Batch Size**: proses Adalah suatu nilai integer yang menentukan berapa banyak sampel yang diproses di dalam jaringan saraf atau neural network dalam satu waktu. <br> <br>
    """, unsafe_allow_html=True)
    new_line()
    
    # Source Code
    st.header("📂 Source Code")
    st.markdown("Untuk pengembangan aplikasi ini, sourcecode tersedia di [**GitHub**](https://github.com/hayuraaa/Forecast-LSTM-GRU.git). Jangan ragu untuk berkontribusi, memberikan feedback, atau menyesuaikan aplikasi agar sesuai dengan kebutuhan Anda.", unsafe_allow_html=True)
    new_line()
    
    # Contributors
    st.header(" 👤 Contributors")
    st.markdown("Aplikasi ini dibuat dan dibangun oleh **Haris Yunanda Rangkuti**(200170154) untuk kebutuhan tugas akhir/skripsi.", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""Jika anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi **yunandaharis@gmail.com**. We're here to help!
    
**Connect with us on social media:** 

<a href="https://www.linkedin.com/in/harisyunanda20/" target="_blank">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe0adDoUGWVD3jGzfT8grK5Uhw0dLXSk3OWJwZaXI-t95suRZQ-wPF7-Az6KurXDVktV4&usqp=CAU" alt="LinkedIn" width="80" height="80" style="border-radius: 25%;">
</a>  󠁪 󠁪 󠁪 󠁪 󠁪 
<a href="https://www.instagram.com/harisrangkuti_/" target="_blank">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram" width="80" height="80" style="border-radius: 25%;">
</a>  󠁪 󠁪 󠁪 󠁪 󠁪 
<a href="https://github.com/hayuraaa/" target="_blank">
  <img src="https://seeklogo.com/images/G/github-logo-5F384D0265-seeklogo.com.png" alt="Github" width="80" height="80" style="border-radius: 25%;">
</a>

<br> 
<br>

I look forward to hearing from you and supporting you on your machine learning journey!

    
    """, unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()

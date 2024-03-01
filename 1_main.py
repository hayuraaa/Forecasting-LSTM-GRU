import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():

    # Dataframe selection
    st.markdown("<h1 align='center'> <b> Sistem Prediksi Cryptocurrency", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Sistem Prediksi ini menggunakan Algoritma Long Short Term Memory dan Gated Recurrent Unit, sistem ini diharapkan dapat menjadi landasan untuk strategi perdagangan yang lebih cerdas dan keputusan investasi yang lebih baik dalam cryptocurrency.", unsafe_allow_html=True)
    new_line(1)
    
    st.divider()
    
    # Dataframe selection
    st.markdown("<h2 align='center'> <b> Getting Started", unsafe_allow_html=True)
    new_line(1)
    st.write(".")
    new_line(1)

    
    
    
if __name__ == "__main__":
    main()

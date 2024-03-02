import streamlit as st
import pandas as pd
import yfinance as yf

def get_crypto_data(coins):
    # Menyiapkan data untuk ditampilkan dalam tabel
    crypto_list = []
    for coin in coins:
        crypto_info = yf.Ticker(coin)
        if crypto_info:
            crypto_name = crypto_info.info['longName']
            crypto_symbol = coin
            
            # Mendapatkan harga koin kripto
            crypto_price = get_crypto_price(crypto_info)
            
            # Mendapatkan URL gambar ikon koin kripto
            crypto_icon = crypto_info.info.get('logo_url')

            # Tambahkan informasi ke dalam list
            crypto_list.append({
                'Nama': crypto_name,
                'Kode': crypto_symbol,
                'Harga': crypto_price,
            })

    return pd.DataFrame(crypto_list)

def get_crypto_price(crypto_info):
    try:
        # Mendapatkan harga koin kripto dari yfinance
        crypto_data = crypto_info.history(period="1d")
        if not crypto_data.empty:
            return crypto_data['Close'][0]
    except Exception as e:
        print(f"Error getting price for {crypto_info}: {e}")
    return None

def main():
    st.title('Tabel Koin Kripto Ekosistem Metaverse📊')
    st.write("Seiring dengan pesatnya pertumbuhan teknologi, metaverse menjadi sorotan utama, terutama dalam ranah asset kripto. Perkembangan terbaru menunjukkan bagaimana asset kripto semakin terintegrasi dalam ekosistem metaverse, menciptakan peluang baru dan mengubah paradigma cara kita berinteraksi di dunia maya. Semula ekosistem metaverse digunakan oleh para pengembang teknologi untuk simulasi permainan, tetapi saat ini ekosistem metaverse juga dapat melakukan transaksi jual beli dengan menggunakan mata uang kripto hingga kepemilikan non fungible token (NFT) Hal ini menunjukkan bahwa variasi jenis aset kripto digunakan secara fleksibel pada ekosistem metaverse dan salah satu volatilitas yang tinggi dalam ekosistem yang ada pada dunia kripto. Berikut daftar 15 kripto paling populer dalam ekosistem metaverse :")

    # Koin yang ingin ditampilkan dalam tabel
    coins = ['BTC-USD', 'STX4847-USD', 'RNDR-USD', 'ICP-USD', 'AXS-USD', 'WEMIX-USD', 'SAND-USD', 'THETA-USD', 'MANA-USD', 'APE-USD', 'ENJ-USD', 'ZIL-USD','ILV-USD', 'EGLD-USD','MASK8536-USD','SUSHI-USD']

    # Mendapatkan data koin kripto
    crypto_df = get_crypto_data(coins)

    # Menampilkan tabel koin kripto
    st.table(crypto_df)
    
    st.write('Untuk lebih lengkap nya dapat dilihat pada CoinMarketCap https://coinmarketcap.com/view/metaverse/')
    
if __name__ == '__main__':
    main()

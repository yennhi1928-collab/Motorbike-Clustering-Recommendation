import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import unicodedata
import re
import string
from pyvi.ViTokenizer import tokenize
from underthesea import pos_tag
from nltk.tokenize import sent_tokenize
import joblib
import datetime as dt
import nltk

# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="Motorbike Clustering & Recommendation", layout="wide")

# -------------------------
# Utility functions
# -------------------------

def load_dict(file_path):
    d = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            if '\t' in line:
                key, value = line.strip().split('\t', 1)
                d[key.lower()] = value
    return d

emoji_dict = load_dict('files/emojicon.txt')
teen_dict = load_dict('files/teencode.txt')
english_dict = load_dict('files/english-vnmese.txt')

with open('files/wrong-word.txt', 'r', encoding='utf8') as f:
    wrong_lst = set(line.strip() for line in f if line.strip())

with open('files/vietnamese-stopwords.txt', 'r', encoding='utf8') as f:
    stopwords_lst = set(f.read().splitlines())

# -------------------------
# Text preprocessing
# -------------------------
def process_query(text):
    if not isinstance(text, str):
        text = str(text)

    # Normalize Unicode & lowercase
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r'\.+', '.', text)

    # Replace emoji
    def replace_emoji(t):
        new_text = ''
        for sentence in sent_tokenize(t):
            words = sentence.split()
            new_sentence = ' '.join(emoji_dict.get(w, w) for w in words)
            new_text += new_sentence + '. '
        return new_text.strip()

    text = replace_emoji(text)

    # Teencode → normal
    words = [teen_dict.get(w, w) for w in text.split()]
    text = ' '.join(words)

    # Remove wrong words
    words = [w for w in text.split() if w not in wrong_lst]
    text = ' '.join(words)

    # Cleaning
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+|\d{9,11}", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    # English → Vietnamese
    words = [english_dict.get(w.lower(), w) for w in text.split()]
    text = " ".join(words)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    text = tokenize(text)

    # POS tagging → keep N, A, V, R
    tagged = pos_tag(text)
    text = " ".join([w for w, t in tagged if t in ("N", "A", "V", "R")])

    # Remove stopwords
    text = " ".join([w for w in text.split() if w not in stopwords_lst])
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -------------------------
# Load model & data
# -------------------------
@st.cache_resource
def load_gensim_model():
    with open("xe_gensim_sim.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_cluster_model():
    return joblib.load("cluster_pipeline.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("data_motobikes.xlsx")
    # Chuẩn hóa cột
    d = {ord('đ'): 'd', ord('Đ'): 'D'}
    def clean_col(name):
        s = unicodedata.normalize('NFKD', str(name)).translate(d)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r'\W+', '_', s.lower()).strip('_')
    df.columns = [clean_col(c) for c in df.columns]
    return df

gensim_model = load_gensim_model()
cluster_pipeline = load_cluster_model()
df_data = load_data()


# Header container: 2 cột – trái (title + intro), phải (logo + caption)
col_left, col_right = st.columns([5, 1])

with col_left:
    st.markdown(
        "<h1 style='margin:0;'>Motorbike Clustering & Recommendation</h1>",
        unsafe_allow_html=True
    )

with col_right:
    st.image("logo.png", width=150)

# Tạo tabs
tabs = st.tabs([
    "🏠 Introduction",
    "📈 Model Selection",
    "🔍 Motorbike Recommendation",
    "🛵 Motorbike Clustering"
])

# TAB 1 – HOME
with tabs[0]:
    st.title("🏠 Introduction")
    st.write("This system provides **2 main functionalities based on used motorcycle data from Chotot website**:")

    # Columns chính: trái – phải
    col_left, col_right = st.columns([2, 1])  # trái rộng hơn

    # ==== Bên trái: 2 chức năng trên-dưới với màu nền nhẹ ====
    with col_left:
        st.markdown(
            """
            <div style="background-color:#E0F7FA; padding:15px; border-radius:10px; margin-bottom:10px;">
            <h4>🔍 Motorbike Recommendation</h4>
            <p>Find similar motorbikes using a pre-trained Gensim model based on user description or a sample motorbike.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="background-color:#FFF3E0; padding:15px; border-radius:10px; margin-top:10px;">
            <h4>🛵 Motorbike Clustering</h4>
            <p>Automatically assign motorbikes to 3 clusters based on brand, type, engine capacity, year, kilometers traveled, and price.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==== Bên phải: các số liệu tổng quan với màu card nhẹ ====
    with col_right:
        total_bikes = len(df_data)
        total_clusters = len(df_data['cluster'].unique()) if 'cluster' in df_data.columns else 3

        # Custom card style bằng markdown
        st.markdown(f"""
        <div style="background-color:#E8F5E9; padding:15px; border-radius:10px; margin-bottom:10px;">
            <h4>Total Motorbikes</h4>
            <p style="font-size:24px; font-weight:bold;">{total_bikes:,}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color:#E3F2FD; padding:15px; border-radius:10px;">
            <h4>Total Clusters 🟢🔵🟠</h4>
            <p style="font-size:20px; font-weight:bold;">{total_clusters}</p>
            <div style="margin-top:10px;">
            - 🟢 Cluster 0 – Low-priced popular motorcycles<br>
            - 🔵 Cluster 1 – Modern scooters & motorcycles<br>
            - 🟠 Cluster 2 – High-mileage motorcycles
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")  # Ngang chia section
    # Hướng dẫn
    st.info("👉 Switch to the next tab to see the model selection process.")
    # Ảnh minh họa toàn trang
    st.image("xe_may_cu.jpg", use_container_width=True)


# TAB 2 – MODEL SELECTION
with tabs[1]:
    st.title("📈 Model Selection")
    st.write("Summary of Recommendation and Clustering Models:")
    st.image("Summarize.png", use_container_width=True)
    st.write("This product was developed by our team:")
    st.image("team.png", use_container_width=True)

# TAB 3 – RECOMMENDATION USING GENSIM
with tabs[2]:
    st.title("🔍 Motorbike Recommendation")
    
    # Highlight info
    st.info("Select a motorbike from the list or enter a description below, then click the button to find similar motorbikes.")

    df = df_data.copy()

    # Random 10 bikes
    sample_df = df.sample(10, random_state=42).reset_index(drop=True)
    sample_df_display = sample_df.assign(
        content = sample_df['tieu_de'] + " – " + sample_df['mo_ta_chi_tiet']
    )[['id','tieu_de','mo_ta_chi_tiet','content']]

    sample_df_display = sample_df_display.rename(columns={
        "id": "ID",
        "tieu_de": "Title",
        "mo_ta_chi_tiet": "Detailed Description",
        "content": "Content"
    })

    st.subheader("Sample Motorbikes")
    st.dataframe(sample_df_display)

    # Input
    selected_title = st.selectbox("Select a motorbike:", sample_df['tieu_de'].tolist())
    query_text = st.text_area("Or enter a motorbike description:")

    if st.button("🔍 Find Similar Motorbikes"):
        try:
            # Process query
            if selected_title and not query_text.strip():
                item = df[df["tieu_de"] == selected_title].iloc[0]
                original_text = item["mo_ta_chi_tiet"]
                query = process_query(original_text)
                header_text = f"**Selected Motorbike:** {item['tieu_de']}"
            else:
                original_text = query_text.strip()
                query = process_query(original_text)
                header_text = "**Entered Description:**"

            # Convert query → BOW
            tokens = query.split()
            query_bow = gensim_model["dictionary"].doc2bow(tokens)

            # Calculate similarity
            sims = np.array(gensim_model["index"][gensim_model["tfidf"][query_bow]]).flatten()

            # Get top 3 indexes (exclude itself)
            top_idx = sims.argsort()[::-1][1:4]

            # Map index → original ID
            top_ids = [gensim_model["id_map"][i] for i in top_idx]

            # Get DataFrame in top 3 order
            top3_df = df.set_index('id').loc[top_ids].reset_index()
            top3_df = top3_df.assign(
                content=top3_df['tieu_de'] + " – " + top3_df['mo_ta_chi_tiet']
            )[['id', 'tieu_de', 'mo_ta_chi_tiet', 'content']]

            top3_df = top3_df.rename(columns={
                "id": "ID",
                "tieu_de": "Title",
                "mo_ta_chi_tiet": "Detailed Description",
                "content": "Content"
            })

            # -----------------------------
            # 1️⃣ SHOW INPUT MOTORBIKE / QUERY
            # -----------------------------
            st.subheader("📝 Your Input Motorbike / Query")
            st.markdown(
                "<div style='background-color:#fff4e6; padding:12px; border-radius:6px;'>"
                f"<b>{header_text}</b><br>{original_text}"
                "</div>",
                unsafe_allow_html=True
            )

            # -----------------------------
            # 2️⃣ SHOW TOP 3 SIMILAR
            # -----------------------------
            st.subheader("🔎 Top 3 Similar Motorbikes")
            with st.container():
                st.markdown(
                    "<div style='background-color:#f0f8ff; padding:10px; border-radius:6px;'>"
                    f"{top3_df.to_html(index=False)}"
                    "</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {e}")
        
# TAB 4 – CLUSTERING
with tabs[3]:
    st.title("🛵 Motorbike Clustering")
    
    st.info("Enter motorbike information below to predict which cluster it belongs to:")

    # --- Mapping cluster -> name & characteristics
    cluster_info = {
        0: "🟢 Cluster – Low-priced popular motorcycles",
        1: "🔵 Cluster – Modern scooters & motorcycles",
        2: "🟠 Cluster – High-mileage motorcycles"
    }

    # --- Category preprocessing same as training
    threshold = 10
    popular_brands = df['thuong_hieu'].value_counts()
    popular_brands = popular_brands[popular_brands >= threshold].index.tolist()

    def preprocess_categories(df_input):
        df_input = df_input.copy()
        # Brand
        df_input['thuong_hieu_proc'] = df_input['thuong_hieu'].apply(lambda x: x if x in popular_brands else 'Other Brand')
        # Engine capacity
        df_input['dung_tich_xe_proc'] = df_input['dung_tich_xe'].replace({
            'Không biết rõ': 'Other',
            'Đang cập nhật': 'Other',
            'Nhật Bản': 'Other'
        })
        # Origin
        df_input['xuat_xu_proc'] = df_input['xuat_xu'].replace('Bảo hành hãng', 'Updating')
        return df_input

    df_proc = preprocess_categories(df)

    # --- Dropdown values from processed data
    brands = sorted(df_proc['thuong_hieu_proc'].dropna().unique())
    loais = sorted(df_proc['loai_xe'].dropna().unique())
    dt_xes = sorted(df_proc['dung_tich_xe_proc'].dropna().unique())
    xuat_xus = sorted(df_proc['xuat_xu_proc'].dropna().unique())

    # --- Input fields in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", brands)
        loai = st.selectbox("Type", loais)

    with col2:
        year = st.number_input("Year of Registration", min_value=1979, max_value=dt.date.today().year, value=2015)
        km = st.number_input("Kilometers Travelled", min_value=0, max_value=1200000, value=30000)

    with col3:
        dt_xe = st.selectbox("Engine Capacity", dt_xes)
        xuat_xu = st.selectbox("Origin", xuat_xus)
        gia_xe = st.number_input("Price (million VND)", min_value=1, max_value=435, value=20)

    # --- Predict cluster
    if st.button("Predict Cluster"):
        df_input = pd.DataFrame([{
            "Brand": brand,
            "Type": loai,
            "Engine Capacity": dt_xe,
            "Origin": xuat_xu,
            "Kilometers Travelled": km,
            "Bike Age": dt.date.today().year - year,
            "Price (million VND)": gia_xe
        }])

        try:
            pred = cluster_pipeline.predict(df_input.rename(columns={
                "Brand": "thuong_hieu",
                "Type": "loai_xe",
                "Engine Capacity": "dung_tich_xe",
                "Origin": "xuat_xu",
                "Kilometers Travelled": "so_km_da_di",
                "Bike Age": "tuoi_xe",
                "Price (million VND)": "gia"
            }))[0]

            # ✅ Show user input information first
            st.subheader("📋 Your Input Information")
            st.markdown(
                """
                <div style='background-color:#f0f8ff; padding:15px; border-radius:8px;'>
                """,
                unsafe_allow_html=True
            )
            st.table(df_input)
            st.markdown("</div>", unsafe_allow_html=True)

            # ✅ Then show cluster result
            st.subheader("🔍 Cluster Prediction Result")

            if pred == 0:
                st.success(f"👉 The motorbike belongs to **{cluster_info[pred]}**")
            elif pred == 1:
                st.info(f"👉 The motorbike belongs to **{cluster_info[pred]}**")
            else:
                st.warning(f"👉 The motorbike belongs to **{cluster_info[pred]}**")

        except Exception as e:
            st.error(f"Error predicting cluster: {e}")

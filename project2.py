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
import os

# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="Phân Cụm & Gợi Ý Xe Máy", layout="wide")
st.image("xe_may_cu.png", use_container_width=True)

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
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("’", "'")
    text = re.sub(r'\.+', '.', text)

    # Replace emoji
    text = ' '.join(emoji_dict.get(w, w) for w in text.split())
    # Teencode → chuẩn
    text = ' '.join(teen_dict.get(w, w) for w in text.split())
    # Remove wrong words
    text = ' '.join([w for w in text.split() if w not in wrong_lst])
    # Cleaning
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+|\d{9,11}", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    # English → Vietnamese
    text = ' '.join([english_dict.get(w.lower(), w) for w in text.split()])
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize & POS tagging
    text = tokenize(text)
    tagged = pos_tag(text)
    text = " ".join([w for w, t in tagged if t in ("N", "A", "V", "R")])
    # Remove stopwords
    text = " ".join([w for w in text.split() if w not in stopwords_lst])
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

# -------------------------
# Header container
# -------------------------
col_left, col_right = st.columns([5, 1])
with col_left:
    st.markdown("<h1 style='margin:0; color:#ffb400;'>Phân Cụm & Gợi Ý Xe Máy</h1>", unsafe_allow_html=True)
    st.markdown(
    "<p style='margin:5px 0 0 0; font-size:14px; font-weight:bold; color:#000; "
    "text-align:left; white-space:nowrap; font-style:italic;'>"
    "Sản phẩm của nhóm Xuân Mai & Yến Nhi, ngày 29/11/2025</p>",
    unsafe_allow_html=True
)
with col_right:
    st.image("logo.png", width=150)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "🏠 Giới Thiệu",
    "📈 Chọn Mô Hình",
    "🔍 Gợi Ý Xe Máy",
    "🛵 Phân Cụm Xe Máy"
])

# -------------------------
# TAB 1 – HOME
# -------------------------
with tabs[0]:
    st.title("🏠 Giới Thiệu")
    st.write("Hệ thống cung cấp **2 chức năng chính dựa trên dữ liệu xe máy cũ từ Chợ Tốt**:")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        <div style="background-color:#ffffff; padding:20px; border-radius:12px; border:1px solid #000; 
                    margin-bottom:15px;">
            <h4 style="color:#ff9800;">🔍 Gợi Ý Xe Máy</h4>
            <p>Tìm những xe máy tương tự dựa trên mô tả của người dùng hoặc xe mẫu bằng mô hình Gensim.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color:#ffffff; padding:20px; border-radius:12px; border:1px solid #000;">
            <h4 style="color:#ff9800;">🛵 Phân Cụm Xe Máy</h4>
            <p>Tự động phân loại xe máy thành 3 cụm dựa trên thương hiệu, loại xe, dung tích, năm sản xuất, km đã đi, và giá bán.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        total_bikes = len(df_data)
        total_clusters = len(df_data['cluster'].unique()) if 'cluster' in df_data.columns else 3

        st.markdown(f"""
        <div style="background-color:#ffffff; padding:15px 20px; border-radius:12px; border:1px solid #000; 
                    text-align:left; margin-bottom:15px;">
            <h4 style="color:#ff9800; margin:0 0 5px 0;">Tổng số xe</h4>
            <p style="font-size:24px; font-weight:bold; margin:0;">{total_bikes:,}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color:#ffffff; padding:15px 20px; border-radius:12px; border:1px solid #000; 
                    text-align:left;">
            <h4 style="color:#ff9800;">Tổng số cụm 🟢🔵🟠</h4>
            <p style="font-size:20px; font-weight:bold;">{total_clusters}</p>
            <ul style="margin:5px 0 0 20px; padding:0;">
                <li>🟢 Cụm 0 – Xe phổ biến giá thấp</li>
                <li>🔵 Cụm 1 – Xe phổ thông, ít km</li>
                <li>🟠 Cụm 2 – Xe mới cao cấp</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background-color:#ffffff; color:#000; padding:12px; border-radius:12px; font-size:16px; margin-bottom:15px;">
        👉 Chuyển sang tab tiếp theo để xem quá trình chọn mô hình.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# TAB 2 – MODEL SELECTION
# -------------------------
with tabs[1]:
    st.title("📈 Chọn Mô Hình")
    st.write("Tóm tắt các mô hình Gợi Ý và Phân Cụm:")
    st.image("Summarize.png", use_container_width=True)
    st.write("Sản phẩm được phát triển bởi nhóm chúng tôi:")
    st.image("team.png", use_container_width=True)

# -------------------------
# TAB 3 – RECOMMENDATION
# -------------------------
with tabs[2]:
    st.title("🔍 Gợi Ý Xe Máy")
    st.markdown("""
    <div style="background-color:#ffffff; color:#000; padding:15px; border-radius:12px; font-size:16px;">
        🔹 Chọn một xe mẫu hoặc nhập mô tả, sau đó nhấn nút để tìm các xe tương tự.
    </div>
    """, unsafe_allow_html=True)

    df = df_data.copy()
    sample_df = df.sample(10, random_state=42).reset_index(drop=True)
    sample_df_display = sample_df.assign(
        content=sample_df['tieu_de'] + " – " + sample_df['mo_ta_chi_tiet']
    )[['id','tieu_de','mo_ta_chi_tiet','content']]

    sample_df_display = sample_df_display.rename(columns={
        "id": "ID",
        "tieu_de": "Tiêu Đề",
        "mo_ta_chi_tiet": "Mô Tả Chi Tiết",
        "content": "Nội Dung"
    })

    st.subheader("Các Xe Mẫu")
    st.dataframe(sample_df_display)

    selected_title = st.selectbox("Chọn một xe:", sample_df['tieu_de'].tolist())
    query_text = st.text_area("Hoặc nhập mô tả xe:")

    if st.button("🔍 Tìm Xe Tương Tự"):
        try:
            if selected_title and not query_text.strip():
                item = df[df["tieu_de"] == selected_title].iloc[0]
                original_text = item["mo_ta_chi_tiet"]
                query = process_query(original_text)
                header_text = f"**Xe đã chọn:** {item['tieu_de']}"
            else:
                original_text = query_text.strip()
                query = process_query(original_text)
                header_text = "**Mô tả nhập vào:**"

            tokens = query.split()
            query_bow = gensim_model["dictionary"].doc2bow(tokens)
            sims = np.array(gensim_model["index"][gensim_model["tfidf"][query_bow]]).flatten()
            top_idx = sims.argsort()[::-1][1:4]
            top_ids = [gensim_model["id_map"][i] for i in top_idx]
            top3_df = df.set_index('id').loc[top_ids].reset_index()
            top3_df = top3_df.assign(
                content=top3_df['tieu_de'] + " – " + top3_df['mo_ta_chi_tiet']
            )[['id', 'tieu_de', 'mo_ta_chi_tiet', 'content']]

            top3_df = top3_df.rename(columns={
                "id": "ID",
                "tieu_de": "Tiêu Đề",
                "mo_ta_chi_tiet": "Mô Tả Chi Tiết",
                "content": "Nội Dung"
            })

            st.subheader("📝 Thông Tin Xe / Mô Tả")
            st.markdown(
                f"<div style='background-color:#ffffff; padding:15px; border-radius:12px;'>"
                f"<b>{header_text}</b><br>{original_text}</div>",
                unsafe_allow_html=True
            )

            st.subheader("🔎 Top 3 Xe Tương Tự")
            st.markdown(
                f"<div style='background-color:#ffffff; padding:15px; border-radius:12px;'>"
                f"{top3_df.to_html(index=False)}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Lỗi: {e}")

# -------------------------
# TAB 4 – CLUSTERING
# -------------------------
with tabs[3]:
    st.title("🛵 Phân Cụm Xe Máy")
    st.markdown("""
    <div style="background-color:#ffffff; color:#000; padding:15px; border-radius:12px; font-size:16px;">
        🔹 Nhập thông tin xe để dự đoán cụm thuộc về:
    </div>
    """, unsafe_allow_html=True)

    cluster_info = {
        0: {"icon": "🟢", "name": "Cụm – Xe phổ biến giá thấp", "desc": "Xe cũ, chạy nhiều km, giá thấp, chủ yếu Honda/Yamaha, tay ga/xe số phổ thông."},
        1: {"icon": "🔵", "name": "Cụm – Xe phổ thông, ít km", "desc": "Xe phổ thông, ít km, giá thấp–trung bình, đa dạng loại và xuất xứ, chủ yếu Honda/Yamaha."},
        2: {"icon": "🟠", "name": "Cụm – Xe mới cao cấp", "desc": "Xe mới, ít km, giá cao, chủ yếu tay ga và tay côn, thương hiệu phổ biến nhưng có xe cao cấp, xuất xứ đa dạng."}
    }

    threshold = 10
    popular_brands = df['thuong_hieu'].value_counts()
    popular_brands = popular_brands[popular_brands >= threshold].index.tolist()

    def preprocess_categories(df_input):
        df_input = df_input.copy()
        df_input['thuong_hieu_proc'] = df_input['thuong_hieu'].apply(lambda x: x if x in popular_brands else 'Other Brand')
        df_input['dung_tich_xe_proc'] = df_input['dung_tich_xe'].replace({'Không biết rõ': 'Other','Đang cập nhật': 'Other','Nhật Bản':'Other'})
        df_input['xuat_xu_proc'] = df_input['xuat_xu'].replace('Bảo hành hãng','Updating')
        return df_input

    df_proc = preprocess_categories(df)

    brands = sorted(df_proc['thuong_hieu_proc'].dropna().unique())
    loais = sorted(df_proc['loai_xe'].dropna().unique())
    dt_xes = sorted(df_proc['dung_tich_xe_proc'].dropna().unique())
    xuat_xus = sorted(df_proc['xuat_xu_proc'].dropna().unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.selectbox("Thương Hiệu", brands)
        loai = st.selectbox("Loại Xe", loais)
    with col2:
        year = st.number_input("Năm đăng ký", min_value=1979, max_value=dt.date.today().year, value=2015)
        km = st.number_input("Số km đã đi", min_value=0, max_value=1200000, value=30000, step=500)
    with col3:
        dt_xe = st.selectbox("Dung tích", dt_xes)
        xuat_xu = st.selectbox("Xuất xứ", xuat_xus)
        gia_xe = st.number_input("Giá (triệu VND)", min_value=1, max_value=435, value=20)

    if st.button("🔍 Dự Đoán Cụm"):
        df_input = pd.DataFrame([{
            "Brand": brand, "Type": loai, "Engine Capacity": dt_xe,
            "Origin": xuat_xu, "Kilometers Travelled": km,
            "Bike Age": dt.date.today().year - year,
            "Price (million VND)": gia_xe
        }])
        try:
            pred = cluster_pipeline.predict(df_input.rename(columns={
                "Brand":"thuong_hieu","Type":"loai_xe","Engine Capacity":"dung_tich_xe",
                "Origin":"xuat_xu","Kilometers Travelled":"so_km_da_di",
                "Bike Age":"tuoi_xe","Price (million VND)":"gia"
            }))[0]

            st.subheader("📋 Thông Tin Xe Nhập Vào")
            st.markdown("<div style='background-color:#ffffff; padding:15px; border-radius:12px;'>", unsafe_allow_html=True)
            st.table(df_input)
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("🔍 Kết Quả Phân Cụm")
            info = cluster_info[pred]
            st.markdown(
                f"<div style='background-color:#ffffff; padding:20px; border-radius:12px;'>"
                f"<h3 style='color:#ff9800;'>{info['icon']} {info['name']}</h3>"
                f"<p style='font-size:16px;'>{info['desc']}</p></div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Lỗi dự đoán cụm: {e}")

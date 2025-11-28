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

# Page config
# -------------------------
st.set_page_config(page_title="Gợi Ý & Phân Cụm Xe Máy", layout="wide")
st.image("banner.png", use_container_width=True, width=100)
# -------------------------
# Utility: load data/models
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data_motobikes.xlsx")
    # Chuẩn hóa tên cột
    d = {ord('đ'): 'd', ord('Đ'): 'D'}
    def clean_col(name):
        s = unicodedata.normalize('NFKD', str(name)).translate(d)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r'\W+', '_', s.lower()).strip('_')
    df.columns = [clean_col(c) for c in df.columns]
    return df

@st.cache_resource
def load_gensim_model():
    with open("xe_gensim_sim.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_cluster_model():
    return joblib.load("cluster_pipeline.pkl")

df_data = load_data()
gensim_model = load_gensim_model()
cluster_pipeline = load_cluster_model()

# -------------------------
# Text preprocessing
# -------------------------
def process_query(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = tokenize(text)
    tagged = pos_tag(text)
    text = " ".join([w for w, t in tagged if t in ("N", "A", "V", "R")])
    return text

# -------------------------
# Header container
# -------------------------
col_left, col_right = st.columns([5, 1])
with col_left:
    st.markdown(
        "<p style='margin:5px 0 0 0; font-size:14px; color:#333333; "
        "text-align:left; white-space:nowrap; font-style:italic;'>"
        "Sản phẩm của Xuân Mai & Yến Nhi, ngày 29/11/2025</p>",
        unsafe_allow_html=True
    )
st.markdown("---")
# ==========================
# Khởi tạo key cho radio nếu chưa có
if 'menu' not in st.session_state:
    st.session_state['menu'] = "🏠 Trang Chủ"

if 'menu_radio' not in st.session_state:
    st.session_state['menu_radio'] = st.session_state['menu']

# Callback để thay đổi menu
def set_menu():
    st.session_state['menu'] = st.session_state['menu_radio']

# Sidebar radio
st.sidebar.radio(
    "📂 Danh Mục:",
    ["🏠 Trang Chủ", "📈 Chọn Mô Hình", "🔍 Gợi Ý Xe Máy", "🛵 Phân Cụm Xe Máy"],
    index=["🏠 Trang Chủ", "📈 Chọn Mô Hình", "🔍 Gợi Ý Xe Máy", "🛵 Phân Cụm Xe Máy"].index(st.session_state['menu_radio']),
    key='menu_radio',
    on_change=set_menu
)

# ==========================
# TAB 1 – Giới Thiệu
# ==========================
if st.session_state['menu'] == "🏠 Trang Chủ":
    st.title("🏠 Trang Chủ")
    st.write("Hệ thống cung cấp **2 chức năng chính**:")
    def go_to_goi_y_tab():
        st.session_state['menu'] = "🔍 Gợi Ý Xe Máy"

    def go_to_phan_cum_tab():
        st.session_state['menu'] = "🛵 Phân Cụm Xe Máy"

    col1, col2 = st.columns(2)

    with col1:
        st.button("🔍 Gợi Ý Xe Máy → ", on_click=go_to_goi_y_tab)
        st.markdown("""
        <div style="
            padding:20px; background-color:#f0f0f0; border-radius:14px;
            box-shadow:0 4px 12px rgba(0,0,0,0.2); height:300px;
            transition: transform 0.2s; cursor:pointer;
        ">
            <div style="font-size:18px; font-weight:bold;">🔍 Gợi Ý Xe Máy</div>
            <div>Hệ thống giúp bạn tìm xe máy tương tự dựa vào thông tin được nhập hoặc chọn thông tin xe mẫu có sẵn</div>
            <hr>
            <div style="font-size:18px; font-weight:bold;">📌 Tổng Số Xe</div>
            <div>7,208 xe</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.button("🛵 Phân Cụm Xe Máy → ", on_click=go_to_phan_cum_tab)
        st.markdown("""
        <div style="
            padding:20px; background-color:#f0f0f0; border-radius:14px;
            box-shadow:0 4px 12px rgba(0,0,0,0.2); height:300px;
            transition: transform 0.2s; cursor:pointer;
        ">
            <div style="font-size:18px; font-weight:bold;">🛵 Phân Cụm Xe Máy</div>
            <div>Từ dữ liệu xe máy bạn nhập vào phân thành 1 trong 3 nhóm đặc trưng</div>
            <hr>
            <ul style="margin:0; padding-left:20px;">
                <li>🟢 Phân khúc phổ thông</li>
                <li>🔵 Phân khúc cao cấp</li>
                <li>🟠 Phân khúc giá rẻ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
# ==========================
# TAB 2 – Chọn Mô Hình
# ==========================
elif st.session_state['menu'] == "📈 Chọn Mô Hình":
    st.title("📈 Chọn Mô Hình")
    st.write("Tóm tắt các mô hình Gợi Ý và Phân Cụm:")
    st.image("Summarize.png", use_container_width=True)
    st.write("Sản phẩm được phát triển bởi nhóm chúng tôi:")
    st.image("team.png", use_container_width=True)

# ==========================
# TAB 3 – Gợi Ý Xe Máy
# ==========================
elif st.session_state['menu'] == "🔍 Gợi Ý Xe Máy":
        st.title("🔍 Gợi Ý Xe Máy")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fff9c4, #ffe0b2); 
            color:#333333; 
            padding:20px; 
            border-radius:15px; 
            font-size:18px; 
            text-align:left;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        ">
            ✨ Chọn một xe mẫu hoặc nhập mô tả, sau đó nhấn nút để tìm các xe tương tự:
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

        st.markdown("<span style='font-size:18px; font-weight:bold; margin-bottom:-5px; display:block;'>Chọn một xe:</span>", unsafe_allow_html=True)
        selected_title = st.selectbox("", sample_df['tieu_de'].tolist())

        st.markdown("<span style='font-size:18px; font-weight:bold; margin-bottom:-5px; display:block;'>Hoặc nhập mô tả xe:</span>", unsafe_allow_html=True)
        query_text = st.text_area("", height=120)

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

# ==========================
# TAB 4 – Phân Cụm Xe Máy
# ==========================
elif st.session_state['menu'] == "🛵 Phân Cụm Xe Máy":
        st.title("🛵 Phân Cụm Xe Máy")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fff9c4, #ffe0b2); 
            color:#333333; 
            padding:20px; 
            border-radius:15px; 
            font-size:18px; 
            text-align:left;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        ">
            ✨ Nhập thông tin xe để dự đoán cụm thuộc về:
        </div>
        """, unsafe_allow_html=True)
        df = df_data.copy()
        cluster_info = {
            0: {"icon": "🟢", "name": "Cụm – Xe thuộc phân khúc tầm trung", "desc": "Xe tương đối mới, đã đi nhiều km, giá trung bình, chủ yếu hãng Yamaha/SYM/ Piaggio, tay ga/xe số phổ thông, xuất xứ Việt Nam"},
            1: {"icon": "🔵", "name": "Cụm – Xe thuộc phân khúc cao cấp", "desc": "Xe mới, phân khối cao giá cao, chủ yếu Honda/Ducati/BMW/Harley Davidson."},
            2: {"icon": "🟠", "name": "Cụm – Xe thuộc phân khúc giá rẻ", "desc": "Xe cũ, đi nhiều km, giá thấp, chủ yếu tay ga, chủ yếu hãng Yamaha/Suzuki, xuất xứ Nhật Bản."}
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

                st.subheader("🛵 Kết Quả Phân Cụm")
                info = cluster_info[pred]
                # Map màu cho từng cụm
                cluster_colors = {
                    0: "#4CAF50",
                    1: "#2196F3",
                    2: "#FF9800"
                }

                # Khi hiển thị kết quả
                st.markdown(
                    f"<div style='background-color:#1e1e1e; padding:20px; border-radius:12px;'>"
                    f"<h3 style='color:{cluster_colors[pred]};'>{info['icon']} {info['name']}</h3>"
                    f"<p style='font-size:16px; color:#ffffff;'>{info['desc']}</p></div>",
                    unsafe_allow_html=True
)
            except Exception as e:
                st.error(f"Lỗi dự đoán cụm: {e}")


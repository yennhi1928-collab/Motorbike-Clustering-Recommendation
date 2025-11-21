import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using menu
st.title("Trung Tâm Tin Học")
st.image("xe_may_cu.jpg", caption="Xe máy cũ")
# Using menu
st.title("Trung Tâm Tin Học")
st.image("xe_may_cu.jpg", caption="Xe máy cũ")
# tạo dataframe mẫu, có 3 cột: Thương hiệu, số lượng xe, Giá trung bình

# Vẽ biểu đồ số lượng xe theo thương hiệu
st.subheader("Biểu đồ số lượng xe theo thương hiệu")
fig, ax = plt.subplots()
sns.barplot(x='Thương hiệu', y='Số lượng xe', data=df, ax=ax)
st.pyplot(fig)

menu = ["Home", "Capstone Project", "Summarize information", "clustering project", "recommendation project"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science]")
    st.write("""### Có 2 function chính trong đồ án:    
    - Topic 1: Hệ thống phân cụm xe máy cũ
    - Topic 2: Hệ thống gợi ý xe máy dựa trên nội dung, phân cụm xe máy
             """)

elif choice == 'Summarize information':
    st.write("##### Tóm tắt thông tin về đồ án")
    st.write('''Với bộ data xe máy cũ, đồ án tập trung vào 2 chức năng chính:
             # ''')
          
elif choice == 'clustering project':
    st.write("##### Gợi ý điều khiển project 1: Dự đoán giá xe máy cũ và phát hiện xe máy bất thường")
    st.write("##### Dữ liệu mẫu")
    # đọc dữ liệu từ file subset_100motobykes.csv
    df = pd.read_csv("subset_100motobikes.csv")
    st.dataframe(df.head())   

    # Trường hợp 2: Đọc dữ liệu từ file csv, excel do người dùng tải lên
    st.write("### Đọc dữ liệu từ file csv do người dùng tải lên")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])   
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dữ liệu đã nhập:")
        st.dataframe(df.head())
    st.write("### 1. Dự đoán giá xe máy cũ")
    # Tạo điều khiển để người dùng nhập các thông tin về xe máy
    thuong_hieu = st.selectbox("Chọn hãng xe", df['Thương hiệu'].unique())
    dong_xe = st.selectbox("Chọn dòng xe", df['Dòng xe'].unique())
    tinh_trang = st.selectbox("Chọn tình trạng", df['Tình trạng'].unique())
    loai_xe = st.selectbox("Chọn loại xe", df['Loại xe'].unique())
    dung_tich_xi_lanh = st.selectbox("Dung tích xi lanh (cc)", df['Dung tích xe'].unique())
    nam_dang_ky = st.slider("Năm đăng ký", 2000, 2024, 2015)
    so_km_da_di = st.number_input("Số km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
    du_doan_gia = st.button("Dự đoán giá")
    if du_doan_gia:
        # In ra các thông tin đã chọn
        st.write("Hãng xe:", thuong_hieu)
        st.write("Dòng xe:", dong_xe)
        st.write("Tình trạng:", tinh_trang)
        st.write("Loại xe:", loai_xe)
        st.write("Dung tích xi lanh (cc):", dung_tich_xi_lanh)
        st.write("Năm đăng ký:", nam_dang_ky)
        st.write("Số km đã đi:", so_km_da_di)
        # Giả sử giá dự đoán là 15000000 VND, thực tế cần dùng mô hình ML để dự đoán
        gia_du_doan = 15000000
        st.write("Giá dự đoán (giả sử), thực tế cần dùng mô hình ML để dự đoán:", gia_du_doan)
    # Làm tiếp cho phần phát hiện xe máy bất thường
    st.write("### 2. Phát hiện xe máy bất thường")
    so_km_bat_thuong = st.number_input("Nhập số km đã đi để kiểm tra bất thường", min_value=0, max_value=200000, value=50000, step=1000)
    gia_du_doan = st.number_input("Nhập giá dự đoán (VND) để kiểm tra bất thường", min_value=0, max_value=100000000, value=15000000, step=100000)
    kiem_tra_bat_thuong = st.button("Kiểm tra bất thường")
    if kiem_tra_bat_thuong:
        # In ra các thông tin đã chọn        
        st.write("Số km đã đi:", so_km_bat_thuong)
        st.write("Giá dự đoán (VND):", gia_du_doan)
        # Giả sử nếu số km đã đi > 150000 hoặc giá dự đoán < 5000000 thì là bất thường
        if so_km_bat_thuong > 150000 or gia_du_doan < 5000000:
            st.write("#### Xe máy bất thường")
        else:
            st.write("#### Xe máy bình thường.")
        # Trên thực tế cần dùng mô hình phát hiện bất thường để kiểm tra
        # Nếu có mô hình ML, có thể gọi hàm dự đoán ở đây
        pass

elif choice=='Gợi ý điều khiển project 2':
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")
    # Tạo dataframe có 3 cột là id, title, description
    # Đọc dữ liệu từ file mau_xe_may.xlsx
    df = pd.read_excel("mau_xe_may.xlsx")    
    st.dataframe(df)
    st.write("### 1. Tìm kiếm xe tương tự")
    # Tạo điều khiển để người dùng chọn công ty
    selected_bike = st.selectbox("Chọn xe", df['title'])
    st.write("Xe đã chọn:", selected_bike) 
    # Từ xe đã chọn này, người dùng có thể xem thông tin chi tiết của xe
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm xe dựa trên thông tin người dùng nhập
    search = st.text_input("Nhập thông tin tìm kiếm")
    # Tìm kiếm xe dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    # Trên thực tế sử dụng content-based filtering (cosine similarity/ gensim) để tìm kiếm xe tương tự
    result = df[df['title'].str.lower().str.contains(search.lower())]    
    # tạo button submit
    tim_kiem = st.button("Tìm kiếm")
    if tim_kiem:
        st.write("Danh sách xe tìm được:")
        st.dataframe(result)
       
# Done
    
    
    
        

        
        

    




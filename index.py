# =========================================
# 1. IMPORT THƯ VIỆN
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

plt.style.use("default")

# =========================================
# 2. ĐỌC DỮ LIỆU
# =========================================

df = pd.read_excel("Book1.xlsx", engine="openpyxl")

df.columns = ["Year", "CO2", "Temperature", "IceExtent"]

df.head()

# ==========================================
#3. KIỂM TRA DỮ LIỆU
# ==========================================

# ---------- 3.1 Kiểm tra kích thước dữ liệu ----------

print("="*60)
print("KÍCH THƯỚC DỮ LIỆU")
print("="*60)

print(df.shape) #--- kiểm tra kích thước

print()

# ---------- 3.2 Kiểm tra kiểu dữ liệu ----------

print("="*60)
print("KIỂU DỮ LIỆU")
print("="*60)

print(df.dtypes) #--- kiểm tra kiểu dữ liệu

print()

# ---------- 3.3 Kiểm tra giá trị khuyết ----------

print("="*60)
print("GIÁ TRỊ KHUYẾT")
print("="*60)

print(df.isnull().sum()) #--- kiểm tra dữ liệu khuyết

print()

# ---------- 3.4 Kiểm tra dữ liệu trùng lặp ----------

print("="*60)
print("DỮ LIỆU TRÙNG LẶP")
print("="*60)

print(df.duplicated().sum()) #--- kiểm tra dữ liệu trùng

# ==========================================
# BƯỚC 4. THỐNG KÊ MÔ TẢ DỮ LIỆU
# ==========================================

desc = df.describe().round(3)

print(desc)

# ==========================================
# BƯỚC 5. PHÂN TÍCH THĂM DÒ DỮ LIỆU (EDA)
# ==========================================

# ---------- 5.1 Biểu đồ xu hướng nồng độ CO₂ ----------

plt.figure(figsize=(12,5))

plt.plot(
    df["Year"],
    df["CO2"],
    marker="o",
    linewidth=2.5,
    color="forestgreen"
)

plt.title("Xu hướng nồng độ CO₂ trong khí quyển giai đoạn 1980–2024")

plt.xlabel("Năm")

plt.ylabel("Nồng độ CO₂ (ppm)")

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("figures/Hinh_4_1_CO2.png", dpi=300)

plt.show()

# ==========================================
# ---------- 5.2 Biểu đồ xu hướng nhiệt độ toàn cầu ----------
# ==========================================

plt.figure(figsize=(12,5))

plt.plot(
    df["Year"],
    df["Temperature"],
    color="red",
    linewidth=2.5,
    marker="o",
    markersize=5
)

plt.title(
    "Xu hướng nhiệt độ trung bình toàn cầu giai đoạn 1980–2024",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Năm", fontsize=12)

plt.ylabel("Sai lệch nhiệt độ (°C)", fontsize=12)

plt.xticks(range(1980,2025,5))

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_2_NhietDo.png",
    dpi=300
)

plt.show()

# ==========================================
# ---------- 5.3 Biểu đồ xu hướng diện tích băng biển ----------
# ==========================================

plt.figure(figsize=(12,5))

plt.plot(
    df["Year"],
    df["IceExtent"],
    color="royalblue",
    linewidth=2.5,
    marker="o",
    markersize=5
)

plt.title(
    "Xu hướng diện tích băng biển Bắc Cực giai đoạn 1980–2024",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Năm", fontsize=12)

plt.ylabel("Diện tích băng biển (triệu km²)", fontsize=12)

plt.xticks(range(1980,2025,5))

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_3_BangBien.png",
    dpi=300
)

plt.show()

# ==========================================
# BƯỚC 6. PHÂN TÍCH TƯƠNG QUAN
# ==========================================

# ---------- 6.1 Tính hệ số tương quan Pearson ----------

corr = df.corr(numeric_only=True)

print(corr.round(3))

# ==========================================
# ---------- 6.2 Trực quan hóa Heatmap ----------
# ==========================================

plt.figure(figsize=(7,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Ma trận tương quan giữa các biến",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_4_Heatmap.png",
    dpi=300
)

plt.show()

# ==========================================
# ---------- 6.3 Trực quan hóa Scatter Plot ----------
# ==========================================

plt.figure(figsize=(7,5))

plt.scatter(
    df["Temperature"],
    df["IceExtent"],
    color="steelblue",
    s=50
)

plt.title(
    "Mối quan hệ giữa nhiệt độ và diện tích băng biển",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Sai lệch nhiệt độ (°C)")
plt.ylabel("Diện tích băng biển (triệu km²)")

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_5_Scatter.png",
    dpi=300
)

plt.show()

# ==========================================
# BƯỚC 7. XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------- 7.1 Chuẩn bị dữ liệu ----------

X = df[["Temperature"]]
y = df["IceExtent"]

# ---------- 7.2 Chia dữ liệu huấn luyện và kiểm tra ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------- 7.3 Huấn luyện mô hình ----------

model_temp = LinearRegression()

model_temp.fit(X_train, y_train)

# ---------- 7.4 In phương trình hồi quy ----------

print("Intercept =", model_temp.intercept_)
print("Coefficient =", model_temp.coef_[0])

# ---------- 7.5 Dự đoán trên tập kiểm tra ----------

y_pred = model_temp.predict(X_test)

# ==========================================
# BƯỚC 8. ĐÁNH GIÁ MÔ HÌNH
# ==========================================


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------- 8.1 Tính các chỉ số đánh giá ----------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)

print("MAE =", round(mae,3))
print("RMSE =", round(rmse,3))
print("R2 =", round(r2,3))

# ---------- 8.2 So sánh giá trị thực tế và dự đoán ----------

comparison = pd.DataFrame({
    "Thực tế": y_test.values,
    "Dự đoán": y_pred
})

print(comparison.head(10))


# ---------- 8.3 Trực quan hóa đường hồi quy ----------

plt.figure(figsize=(7,5))

plt.scatter(
    X,
    y,
    color="steelblue",
    s=50
)

plt.plot(
    X,
    model_temp.predict(X),
    color="red",
    linewidth=2
)

plt.title(
    "Hồi quy tuyến tính giữa nhiệt độ và diện tích băng biển",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Sai lệch nhiệt độ (°C)")
plt.ylabel("Diện tích băng biển (triệu km²)")

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_6_HoiQuyNhietDo.png",
    dpi=300
)

plt.show()

# ============================================================
# BƯỚC 9. DỰ BÁO DIỆN TÍCH BĂNG BIỂN ĐẾN NĂM 2050
# ============================================================

# ---------- 9.1 Xây dựng mô hình dự báo ----------

X_year = df[["Year"]]
y_year = df["IceExtent"]

forecast_model = LinearRegression()

forecast_model.fit(
    X_year,
    y_year
)

# ---------- 9.2 Tạo dữ liệu dự báo giai đoạn 2025–2050 ----------

future = pd.DataFrame({
    "Year": range(2025,2051)
})

future["Forecast_IceExtent"] = forecast_model.predict(future)

print(future)

# ---------- 9.3 Thực hiện dự báo ----------

plt.figure(figsize=(12,5))

plt.plot(
    df["Year"],
    df["IceExtent"],
    marker="o",
    label="Dữ liệu thực tế"
)

plt.plot(
    future["Year"],
    future["Forecast_IceExtent"],
    linestyle="--",
    linewidth=2,
    color="red",
    label="Dự báo"
)

plt.title(
    "Dự báo diện tích băng biển Bắc Cực đến năm 2050",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Năm")
plt.ylabel("Diện tích băng biển (triệu km²)")

plt.legend()

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figures/Hinh_4_7_DuBao2050.png",
    dpi=300
)

plt.show()
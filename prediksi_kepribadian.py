import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- DATA PREPARATION ---
df = pd.read_csv("personality_datasert.csv")

# Encoding kolom kategorikal
le_stage = LabelEncoder()
le_drain = LabelEncoder()
le_target = LabelEncoder()

df["Stage_fear"] = le_stage.fit_transform(df["Stage_fear"])
df["Drained_after_socializing"] = le_drain.fit_transform(df["Drained_after_socializing"])
df["Personality"] = le_target.fit_transform(df["Personality"])  # 0 = Extrovert, 1 = Introvert

X = df.drop("Personality", axis=1)
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Prediksi Kepribadian Mahasiswa", layout="centered")

st.markdown("# 🧠 Prediksi Kepribadian Mahasiswa")
st.markdown("### 💡 Tentang Model Prediksi")
st.write("""
Model ini digunakan untuk memprediksi kecenderungan kepribadian mahasiswa, apakah mereka termasuk ke dalam **Introvert** atau **Extrovert**, 
berdasarkan beberapa faktor perilaku sosial dan kebiasaan sehari-hari.

Model yang digunakan adalah **Random Forest Classifier**.  
Model ini bekerja dengan menganalisis data perilaku seperti waktu menyendiri, ukuran lingkaran pertemanan, dan keaktifan sosial.
""")

# --- FAKTOR YANG BERPENGARUH ---
st.markdown("### 📌 Faktor yang Paling Mempengaruhi Prediksi Kepribadian")

feature_importance = pd.DataFrame({
    "Faktor": X.columns,
    "Pengaruh": model.feature_importances_
}).sort_values(by="Pengaruh", ascending=False)

st.dataframe(feature_importance.reset_index(drop=True))

# --- FORM INPUT UNTUK PREDIKSI ---
st.markdown("### 🧾 Masukkan Data Kamu")

with st.form("form_prediksi"):
    alone = st.slider("Berapa lama kamu biasanya menghabiskan waktu sendirian?", 0, 10, 5)
    stage = st.selectbox("Apakah kamu takut tampil di depan umum?", ["Yes", "No"])
    event = st.slider("Seberapa sering kamu hadir di acara sosial?", 0, 10, 5)
    outside = st.slider("Seberapa sering kamu keluar rumah untuk bersosialisasi?", 0, 10, 5)
    drained = st.selectbox("Apakah kamu merasa lelah setelah sosialisasi?", ["Yes", "No"])
    friends = st.slider("Berapa banyak teman dekat yang kamu miliki?", 0, 20, 5)
    post = st.slider("Seberapa sering kamu posting di media sosial?", 0, 10, 5)

    submit = st.form_submit_button("🔮 Prediksi Kepribadian")

if submit:
    input_df = pd.DataFrame([[
        alone,
        le_stage.transform([stage])[0],
        event,
        outside,
        le_drain.transform([drained])[0],
        friends,
        post
    ]], columns=X.columns)

    hasil = model.predict(input_df)[0]
    hasil_str = le_target.inverse_transform([hasil])[0]

    st.markdown("### 📣 Hasil Prediksi:")
    if hasil_str == "Introvert":
        st.warning("Berdasarkan data yang kamu masukkan, kamu cenderung memiliki kepribadian **Introvert** 😌")
    else:
        st.success("Berdasarkan data yang kamu masukkan, kamu cenderung memiliki kepribadian **Extrovert** 😄")

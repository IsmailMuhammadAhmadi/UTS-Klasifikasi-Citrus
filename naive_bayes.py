import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset
df = pd.read_csv("citrus.csv")

# 2. Pisahkan fitur dan label
X = df.drop("name", axis=1)  
y = df["name"]               

# 3. Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat dan melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Lakukan prediksi terhadap data uji
y_pred = model.predict(X_test)

# 6. Evaluasi model
print("~Hasil Evaluasi Model~")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# 7. Tampilkan hasil klasifikasi
print("\n~Prediksi Hasil Klasifikasi~")
hasil_klasifikasi = pd.DataFrame({
    "Data": range(1, len(y_test)+1),
    "Asli": y_test.values,
    "Prediksi": y_pred
}).reset_index(drop=True)

print(hasil_klasifikasi.head(10).to_string(index=False))
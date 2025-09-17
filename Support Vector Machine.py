import numpy as np
import pandas as pd
import cv2
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =======================
# 1. Load Dataset SVM
# =======================
color_data = pd.read_csv('dataset_SVM_merah_hijau.csv')

X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
print("Akurasi model dataset:", svm_model.score(X_test, y_test) * 100, "%")

# =======================
# 2. Fungsi prediksi 1 piksel
# =======================
def predict_color_with_prob(bgr_pixel):
    rgb = bgr_pixel[::-1].reshape(1, -1)  # ke (R,G,B)
    rgb_scaled = scaler.transform(rgb)
    probas = svm_model.predict_proba(rgb_scaled)[0]
    idx = np.argmax(probas)
    return svm_model.classes_[idx], probas[idx]

# =======================
# 3. Kamera + Bounding Box + Akurasi + Logging
# =======================
cap = cv2.VideoCapture(0)
detections_log = []  # list untuk menyimpan hasil deteksi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask warna dasar (filter cepat)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

        # Prediksi rata-rata warna ROI
        avg_color = roi.mean(axis=(0, 1)).astype(int)
        label, prob = predict_color_with_prob(avg_color)
        acc = prob * 100

        # Gambar bounding box + teks
        box_color = (0, 0, 255) if label == 'merah' else (0, 255, 0)
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(output_frame, f"{label} {acc:.1f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        # ---- Tampilkan ke terminal ----
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Deteksi: {label}, Akurasi: {acc:.1f}%, Posisi: x={x}, y={y}, w={w}, h={h}")

        # Simpan ke list log
        detections_log.append({
            "Waktu": timestamp,
            "Warna": label,
            "Akurasi (%)": round(acc, 1),
            "x": x, "y": y, "w": w, "h": h
        })

    cv2.imshow("Deteksi Merah & Hijau (SVM)", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =======================
# 4. Simpan Log ke Excel
# =======================
if detections_log:
    df_log = pd.DataFrame(detections_log)
    excel_file = "log_deteksi_merah_hijau.xlsx"
    df_log.to_excel(excel_file, index=False)
    print(f"Log deteksi disimpan ke: {excel_file}")
else:
    print("Tidak ada deteksi yang disimpan.")

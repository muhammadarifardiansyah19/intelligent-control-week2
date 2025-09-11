import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# =======================
# 1. Load Dataset
# =======================
color_data = pd.read_csv('color_rgb_dataset.csv')

X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data train & test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =======================
# 2. Train Model KNN
# =======================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Akurasi model dataset:", knn.score(X_test, y_test) * 100, "%")

# =======================
# 3. Fungsi cari warna referensi
# =======================
def get_reference_color(color_name):
    """Ambil rata-rata RGB dari warna yang sesuai di dataset"""
    mask = (y == color_name)
    if np.any(mask):
        return np.mean(X[mask], axis=0)
    return np.array([0, 0, 0])

# =======================
# 4. Integrasi Kamera
# =======================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ukuran kotak sampling
    box_size = 100
    height, width, _ = frame.shape
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Ambil area kotak (ROI)
    roi = frame[y1:y2, x1:x2]

    # Hitung rata-rata warna dalam kotak
    avg_color = roi.mean(axis=(0, 1)).astype(int)  # (B, G, R)
    avg_color_rgb = avg_color[::-1]  # ubah ke (R, G, B)

    # Prediksi warna dengan model KNN
    pixel_scaled = scaler.transform([avg_color_rgb])
    color_pred = knn.predict(pixel_scaled)[0]

    # Ambil warna referensi dari dataset
    ref_rgb = get_reference_color(color_pred)

    # Hitung jarak Euclidean
    distance = np.linalg.norm(avg_color_rgb - ref_rgb)

    # Konversi jarak ke "akurasi persen"
    max_dist = np.sqrt(3 * (255 ** 2))  # jarak maksimal RGB (0,0,0) <-> (255,255,255)
    accuracy_percent = max(0, 100 - (distance / max_dist) * 100)

    # Gambar kotak ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Tambahkan teks hasil prediksi dan akurasi
    cv2.putText(frame, f'Color: {color_pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, f'Accuracy: {accuracy_percent:.2f}%', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Preview warna asli di kotak lingkaran
    circle_center = (120, 160)
    circle_radius = 30
    bgr_color = tuple(map(int, avg_color))  # warna BGR asli dari ROI
    cv2.circle(frame, circle_center, circle_radius, bgr_color, -1)

    # Tampilkan kamera
    cv2.imshow('Color Detection with Accuracy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

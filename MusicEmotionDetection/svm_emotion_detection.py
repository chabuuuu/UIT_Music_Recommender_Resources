import os
import numpy as np
import pandas as pd
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ### 1. Hàm trích xuất đặc trưng từ file audio
def extract_features(file_path):
    try:
        # Tải file audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Trích xuất các đặc trưng
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # MFCC
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)     # Chroma
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)  # Spectral contrast
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)                         # Tempo
        rms = np.mean(librosa.feature.rms(y=y))                                # RMS energy
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))                 # Zero-crossing rate
        
        # Gộp các đặc trưng thành một vector
        features = np.concatenate((mfcc, chroma, spectral_contrast, [tempo, rms, zcr]))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ### 2. Tải dữ liệu và trích xuất đặc trưng
def load_data(data_dir, ratings_file):
    ratings = pd.read_csv(ratings_file)
    X = []  # Danh sách đặc trưng
    y = []  # Danh sách nhãn
    
    for index, row in ratings.iterrows():
        file_name = f"{row['Nro']:03d}.mp3"  # Ví dụ: 001.mp3
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(row['TARGET'])  # Lấy nhãn từ cột TARGET
        else:
            print(f"File not found: {file_path}")
    
    return np.array(X), np.array(y)

# ### 3. Main function để chạy toàn bộ quy trình
def main():
    # Đường dẫn đến thư mục chứa file audio và file ratings
    data_dir = 'path/to/Set2'  # Thay bằng đường dẫn đến thư mục chứa file MP3
    ratings_file = 'path/to/mean_ratings_set2.csv'  # Thay bằng đường dẫn đến file CSV
    
    print("Loading data and extracting features...")
    X, y = load_data(data_dir, ratings_file)
    
    if len(X) == 0 or len(y) == 0:
        print("No data loaded. Please check your dataset directory and files.")
        return
    
    # ### 4. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # ### 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ### 6. Huấn luyện mô hình SVM
    print("Training the SVM model...")
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    clf.fit(X_train, y_train)
    
    # ### 7. Đánh giá mô hình
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ### 8. Dự đoán cảm xúc cho một bài hát mới
    new_song_path = 'path/to/new_song.mp3'  # Thay bằng đường dẫn bài hát mới
    print(f"\nClassifying new song: {new_song_path}")
    new_features = extract_features(new_song_path)
    
    if new_features is not None:
        new_features = scaler.transform([new_features])  # Chuẩn hóa đặc trưng
        predicted_label = clf.predict(new_features)[0]
        print(f"Predicted emotion: {predicted_label}")
    else:
        print("Could not extract features from the new song.")

# ### 9. Chạy chương trình
if __name__ == "__main__":
    main()
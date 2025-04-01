import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import librosa
import os

# Định nghĩa hàm ánh xạ valence và arousal sang nhãn cảm xúc
def map_to_emotion(valence, arousal):
    if valence > 5 and arousal > 5:
        return 'HAPPY'
    elif valence < 5 and arousal < 5:
        return 'SAD'
    elif valence > 5 and arousal > 7:
        return 'ENERGY'
    elif valence > 5 and 3 < arousal < 7:
        return 'ROMANTIC'
    elif 3 < valence < 7 and arousal < 3:
        return 'CHILL'
    else:
        return 'OTHER'

# Hàm trích xuất đặc trưng từ file MP3
def extract_features_from_mp3(mp3_path):
    try:
        # Tải file audio
        y, sr = librosa.load(mp3_path, sr=None)
        
        # Trích xuất các đặc trưng
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # 13 đặc trưng
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)     # 12 đặc trưng
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)  # 7 đặc trưng
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]                         # 1 đặc trưng
        rms = np.mean(librosa.feature.rms(y=y))                            # 1 đặc trưng
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))            # 1 đặc trưng
        
        # Gộp các đặc trưng thành một vector
        features = np.concatenate((mfcc, chroma, spectral_contrast, [tempo, rms, zcr]))
        return features  # Tổng cộng 35 đặc trưng
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")
        return None

# Chuẩn bị dữ liệu huấn luyện
def prepare_training_data(annotation_path, audio_dir):
    annotations = pd.read_csv(annotation_path)
    annotations.columns = annotations.columns.str.strip()  # Loại bỏ khoảng trắng trong tên cột

    X = []
    y = []

    for index, row in annotations.iterrows():

        if (index + 1) % 10 == 0:
            print(f"Đang xử lý bài hát {index + 1}/{len(annotations)}: {row['song_id']}")

        song_id = row['song_id']
        mp3_path = os.path.join(audio_dir, f"{int(song_id)}.mp3")  # Giả sử file MP3 có tên theo song_id
        if os.path.exists(mp3_path):
            features = extract_features_from_mp3(mp3_path)
            if features is not None:
                X.append(features)
                emotion = map_to_emotion(row['valence_mean'], row['arousal_mean'])
                y.append(emotion)
        else:
            print(f"File not found: {mp3_path}")

    return np.array(X), np.array(y)

# Huấn luyện mô hình
def train_model(X, y):
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Kích thước tập huấn luyện: {X_train.shape}, Kích thước tập kiểm tra: {X_test.shape}")
    print("Phân bổ các loại cảm xúc trong tập huấn luyện:")
    print(pd.Series(y_train).value_counts())
    print("Phân bổ các loại cảm xúc trong tập kiểm tra:")
    print(pd.Series(y_test).value_counts())

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Huấn luyện mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    print("Đánh giá mô hình trên tập kiểm tra:")
    print(classification_report(y_test, y_pred))

    # Lưu đặc trưng
    np.save('/media/haphuthinh/Data/Workspace/UIT/DO_AN_2/MusicEmotionDetection/TestSong/TrainedModel/random_forest_X_train.npy', X_train)
    np.save('/media/haphuthinh/Data/Workspace/UIT/DO_AN_2/MusicEmotionDetection/TestSong/TrainedModel/random_forest_y_train.npy', y_train)

    return model, scaler

# Hàm dự đoán cảm xúc cho bài hát mới
def predict_emotion(mp3_path, model, scaler):
    features = extract_features_from_mp3(mp3_path)
    if features is not None:
        features = scaler.transform([features])  # Chuẩn hóa đặc trưng
        emotion = model.predict(features)[0]
        return emotion
    else:
        return "Error"

# Thực thi chương trình
if __name__ == "__main__":
    # Đường dẫn đến file annotation và thư mục chứa file MP3
    annotation_path = '/media/haphuthinh/Data/Workspace/UIT/DO_AN_2/MusicEmotionDetection/Dataset/DEAM/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv'  # Thay bằng đường dẫn thực tế
    audio_dir = '/media/haphuthinh/Data/Workspace/UIT/DO_AN_2/MusicEmotionDetection/Dataset/DEAM/MEMD_audio'  # Thay bằng đường dẫn thực tế đến thư mục chứa file MP3

    # Bước 1: Chuẩn bị dữ liệu và huấn luyện mô hình
    print("Đang chuẩn bị dữ liệu huấn luyện...")
    X, y = prepare_training_data(annotation_path, audio_dir)
    if len(X) == 0 or len(y) == 0:
        print("Không tải được dữ liệu. Vui lòng kiểm tra thư mục và file dữ liệu.")
    else:
        print("Đang huấn luyện mô hình...")
        model, scaler = train_model(X, y)

        joblib.dump(model, './TrainedModel/random_forest_emotion_model.pkl')
        joblib.dump(scaler, './TrainedModel/random_forest_scaler.pkl')

        # Bước 2: Nhập đường dẫn bài hát và dự đoán
        mp3_path = "/media/haphuthinh/Data/Workspace/UIT/DO_AN_2/MusicEmotionDetection/TestSong/BacPhanRemix2019-JackG5RDJFuture-6058030.mp3"
        if os.path.exists(mp3_path):
            print("Đang phân tích bài hát...")
            predicted_emotion = predict_emotion(mp3_path, model, scaler)
            print(f"Cảm xúc dự đoán của bài hát là: {predicted_emotion}")
        else:
            print("Đường dẫn không hợp lệ, vui lòng kiểm tra lại!")
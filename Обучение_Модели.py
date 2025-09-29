import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =====================
# КОНФИГУРАЦИЯ
# =====================
DATASET_PATH = "Датасет диагностики"
EXHAUST_PATH = os.path.join(DATASET_PATH, "выхлопная система")
OTHER_PATH = os.path.join(DATASET_PATH, "другие_звуки")
MODEL_PATH = "exhaust_model.h5"
TARGET_SHAPE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# Создаем папки если их нет
os.makedirs(EXHAUST_PATH, exist_ok=True)
os.makedirs(OTHER_PATH, exist_ok=True)

# =====================
# 1. ФУНКЦИИ ДЛЯ ОБРАБОТКИ ЗВУКА
# =====================

def audio_to_spectrogram(audio_path, target_shape=TARGET_SHAPE):
    """Преобразует аудио в спектрограмму"""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
        
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
        
        if spectrogram_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - spectrogram_db.shape[1]
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spectrogram_db = spectrogram_db[:, :target_shape[1]]
        
        return spectrogram_db
    except Exception as e:
        print(f"Ошибка обработки {audio_path}: {e}")
        return None

def load_all_data():
    """Загружает все данные из папок"""
    print("📁 Загружаю данные для обучения...")
    
    X = []
    y = []
    
    # Загружаем звуки выхлопа
    if os.path.exists(EXHAUST_PATH):
        exhaust_files = [f for f in os.listdir(EXHAUST_PATH) if f.endswith(('.wav', '.mp3', '.flac'))]
        for file in exhaust_files:
            audio_path = os.path.join(EXHAUST_PATH, file)
            spectrogram = audio_to_spectrogram(audio_path)
            if spectrogram is not None:
                X.append(spectrogram)
                y.append(1)  # 1 = звук выхлопа
        print(f"🎵 Звуков выхлопа: {len(exhaust_files)}")
    
    # Загружаем другие звуки
    if os.path.exists(OTHER_PATH):
        other_files = [f for f in os.listdir(OTHER_PATH) if f.endswith(('.wav', '.mp3', '.flac'))]
        for file in other_files:
            audio_path = os.path.join(OTHER_PATH, file)
            spectrogram = audio_to_spectrogram(audio_path)
            if spectrogram is not None:
                X.append(spectrogram)
                y.append(0)  # 0 = другие звуки
        print(f"🎵 Других звуков: {len(other_files)}")
    
    if len(X) == 0:
        print("❌ Нет данных для обучения! Добавьте звуки в папки.")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape + (1,))
    
    print(f"📊 Всего примеров: {len(X)}")
    return X, y

def create_model(input_shape):
    """Создает модель нейросети"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# =====================
# 2. ОБУЧЕНИЕ МОДЕЛИ
# =====================

def train_model():
    """Основная функция обучения"""
    X, y = load_all_data()
    
    if X is None:
        return None
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Создаем модель
    model = create_model((TARGET_SHAPE[0], TARGET_SHAPE[1], 1))
    
    print("🤖 Архитектура модели:")
    model.summary()
    
    print("🎯 Начинаю обучение...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
        ]
    )
    
    # Сохраняем модель
    model.save(MODEL_PATH)
    print(f"💾 Модель сохранена как '{MODEL_PATH}'")
    
    # Оценка точности
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Точность на тестовых данных: {test_accuracy:.2%}")
    
    # Визуализация обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("📊 График обучения сохранен как 'training_history.png'")
    
    return model

if __name__ == "__main__":
    print("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 50)
    model = train_model()
    if model:
        print("✅ Обучение завершено успешно!")
    else:
        print("❌ Обучение не удалось!")
    
    input("Нажмите Enter для выхода...")
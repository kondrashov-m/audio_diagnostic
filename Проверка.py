import os
import numpy as np
import librosa
import sounddevice as sd
import wave
import threading
from tensorflow import keras
from pynput import keyboard
import time
import sys
import shutil

# =====================
# КОНФИГУРАЦИЯ
# =====================
DATASET_PATH = "Датасет диагностики"
EXHAUST_PATH = os.path.join(DATASET_PATH, "выхлопная система")
OTHER_PATH = os.path.join(DATASET_PATH, "другие_звуки")
MODEL_PATH = "exhaust_model.h5"
TEMP_AUDIO = "temp_recording.wav"

# Параметры записи
SAMPLE_RATE = 22050
DURATION = 3  # секунды
CHANNELS = 1

# Создаем папки если их нет
os.makedirs(EXHAUST_PATH, exist_ok=True)
os.makedirs(OTHER_PATH, exist_ok=True)

# =====================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================

def load_model():
    """Загружает обученную модель"""
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("✅ Модель загружена успешно")
            return model
        else:
            print("❌ Модель не найдена! Сначала запустите Обучение_Модели.py")
            return None
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None

def record_audio(filename, duration, samplerate):
    """Записывает аудио с микрофона"""
    print(f"🎤 Записываю {duration} секунд...")
    audio_data = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=CHANNELS,
                       dtype='float32')
    sd.wait()  # Ждем окончания записи
    
    # Сохраняем в файл
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    print("✅ Запись сохранена")
    return filename

def audio_to_spectrogram(audio_path, target_shape=(128, 128)):
    """Преобразует аудио в спектрограмму"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=2)
    
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
    
    if spectrogram_db.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram_db = spectrogram_db[:, :target_shape[1]]
    
    return spectrogram_db.reshape(1, target_shape[0], target_shape[1], 1)

def predict_audio(audio_path, model):
    """Определяет тип звука"""
    spectrogram = audio_to_spectrogram(audio_path)
    prediction = model.predict(spectrogram, verbose=0)[0][0]
    
    if prediction > 0.5:
        return f"🔊 ЭТО ЗВУК ВЫХЛОПА! (уверенность: {prediction:.2%})", prediction, 1
    else:
        return f"🚫 Это не звук выхлопа (уверенность: {1-prediction:.2%})", prediction, 0

def save_audio_with_feedback(audio_path, user_feedback, model_prediction):
    """Сохраняет аудио в правильную папку на основе feedback"""
    timestamp = int(time.time())
    new_filename = f"audio_{timestamp}.wav"
    
    if user_feedback == 1:
        # Пользователь согласен с моделью
        if model_prediction == 1:
            save_path = os.path.join(EXHAUST_PATH, new_filename)
        else:
            save_path = os.path.join(OTHER_PATH, new_filename)
    else:
        # Пользователь не согласен с моделью
        if model_prediction == 1:
            save_path = os.path.join(OTHER_PATH, new_filename)  # Модель сказала "выхлоп", но пользователь сказал "нет"
        else:
            save_path = os.path.join(EXHAUST_PATH, new_filename)  # Модель сказала "не выхлоп", но пользователь сказал "да"
    
    # Копируем файл
    shutil.copy(audio_path, save_path)
    print(f"💾 Аудио сохранено в: {save_path}")
    
    return save_path

def retrain_model():
    """Запускает переобучение модели"""
    print("🔄 Запускаю переобучение модели...")
    print("⏳ Данное окно закроется автоматически через 3 секунды...")
    time.sleep(3)
    
    # Закрываем текущее окно и запускаем обучение
    if os.path.exists("Обучение_Модели.py"):
        os.system("python Обучение_Модели.py")
    else:
        print("❌ Файл 'Обучение_Модели.py' не найден!")
    
    sys.exit()

# =====================
# 2. ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ
# =====================

def main():
    """Основная функция"""
    print("🎯 СИСТЕМА ПРОВЕРКИ ЗВУКОВ ВЫХЛОПА")
    print("=" * 40)
    print("Нажмите R - записать звук и проверить")
    print("Нажмите L - переобучить модель")
    print("Нажмите ESC - выход")
    print("=" * 40)
    
    model = load_model()
    if not model:
        return

    # Флаг для блокировки повторного нажатия
    is_processing = False

    def on_press(key):
        nonlocal is_processing
        try:
            if key == keyboard.Key.esc:
                print("\n👋 Выход из программы...")
                return False
            
            elif hasattr(key, 'char') and key.char == 'r' and not is_processing:
                is_processing = True
                print("\n" + "=" * 40)
                
                # Записываем аудио
                record_audio(TEMP_AUDIO, DURATION, SAMPLE_RATE)
                
                # Анализируем
                result, confidence, prediction = predict_audio(TEMP_AUDIO, model)
                print(f"🔍 Результат: {result}")
                
                # Запрашиваем подтверждение
                print("\n❓ Верно ли я определил?")
                if prediction == 1:
                    user_input = input("1 - ДА, это звук выхлопа\n0 - НЕТ, это другой звук\nВаш выбор: ")
                else:
                    user_input = input("1 - ДА, это другой звук\n0 - НЕТ, это звук выхлопа\nВаш выбор: ")
                
                try:
                    feedback = int(user_input)
                    if feedback in [0, 1]:
                        # Сохраняем аудио с учетом feedback
                        save_audio_with_feedback(TEMP_AUDIO, feedback, prediction)
                        print("✅ Спасибо за обратную связь!")
                    else:
                        print("❌ Введите 0 или 1")
                except ValueError:
                    print("❌ Введите число 0 или 1")
                
                print("\nГотов к следующей проверке...")
                is_processing = False
                
            elif hasattr(key, 'char') and key.char == 'l':
                print("\n🔄 Запуск переобучения...")
                # Запускаем в отдельном потоке чтобы не блокировать
                threading.Thread(target=retrain_model, daemon=True).start()
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            is_processing = False
    
    # Запускаем слушатель клавиш с suppress=True
    print("\nОжидание нажатия клавиш...")
    with keyboard.Listener(on_press=on_press, suppress=True) as listener:
        listener.join()

if __name__ == "__main__":
    main()
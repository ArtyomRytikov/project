import cv2
import numpy as np
from tensorflow.keras.models import load_model


def predict_image(model, image_path, charset):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=-1)  # Добавляем размерность канала
    img = img / 255.0

    pred = model.predict(np.array([img]))
    # Декодирование CTC-вывода (например, через greedy-алгоритм)
    text = ""
    for p in pred[0]:
        if p != len(charset):  # Пропускаем blank-символ
            text += charset[int(p)]
    return text
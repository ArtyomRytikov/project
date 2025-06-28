from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense


def build_crnn(input_shape, num_classes):
    """
    Создаёт модель CRNN для OCR.

    Args:
        input_shape (tuple): Размер входного изображения (H, W, C).
        num_classes (int): Количество символов в алфавите + blank.

    Returns:
        Model: Готовая модель Keras.
    """
    # 1. Входной слой (изображение)
    input_img = Input(shape=input_shape, name="image_input")

    # 2. CNN-часть (извлечение признаков)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2))(x)  # Уменьшение размерности в 2 раза

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)  # Ещё одно уменьшение

    # 3. Подготовка данных для LSTM (преобразуем в последовательность)
    x = Reshape((x.shape[1], -1))(x)  # Формат (timesteps, features)

    # 4. RNN-часть (обработка последовательности)
    x = LSTM(128, return_sequences=True)(x)  # Возвращает всю последовательность
    x = LSTM(64, return_sequences=True)(x)

    # 5. Выходной слой (вероятности символов + blank)
    output = Dense(num_classes + 1, activation="softmax", name="ctc_output")(x)

    # Собираем модель
    model = Model(inputs=input_img, outputs=output)
    return model
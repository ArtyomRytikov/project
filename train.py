from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import build_crnn
from data_loader import OCRDataGenerator, load_data

# 1. Загрузка данных
images_paths, texts = load_data("data/labels.txt")  # Пути к изображениям и текстам

# 2. Разделение на тренировочную и валидационную выборку
split_idx = int(0.8 * len(images_paths))
train_images, train_texts = images_paths[:split_idx], texts[:split_idx]
val_images, val_texts = images_paths[split_idx:], texts[split_idx:]

# 3. Создание генераторов данных
train_gen = OCRDataGenerator(
    train_images, train_texts,
    batch_size=32,
    img_size=(128, 32),
    charset=CHARACTERS,
    augment=True
)

val_gen = OCRDataGenerator(
    val_images, val_texts,
    batch_size=32,
    img_size=(128, 32),
    charset=CHARACTERS,
    augment=False
)

# 4. Создание модели
model = build_crnn((32, 128, 1), len(CHARACTERS))

# 5. Компиляция модели (CTC loss)
model.compile(optimizer=Adam(learning_rate=0.001), loss={"ctc_output": lambda y_true, y_pred: y_pred})

# 6. Колбэки (сохранение модели, ранняя остановка)
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True
)

early_stop = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

# 7. Обучение
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=[checkpoint, early_stop],
    verbose=1
)
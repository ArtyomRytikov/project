import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, images_paths, labels, batch_size, img_size, charset):
        self.images_paths = images_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.charset = charset

    def __getitem__(self, idx):
        # Загрузка и предобработка изображений
        batch_images = []
        batch_labels = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            img = cv2.imread(self.images_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
            img = img / 255.0  # Нормализация
            batch_images.append(img)
            # Преобразование текста в числовой формат (для CTC loss)
            batch_labels.append([self.charset.index(c) for c in self.labels[i]])
        return np.array(batch_images), np.array(batch_labels)
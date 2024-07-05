import librosa
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Загрузка аудио файла и вычисление спектрограммы
        y, sr = librosa.load(file_path, sr=None)
        spect = librosa.feature.melspectrogram(y=y, sr=sr)
        spect = librosa.power_to_db(spect, ref=np.max)
        spect = np.expand_dims(spect, axis=0)  # Добавление канала для CNN

        if self.transform:
            spect = self.transform(spect)

        return spect, label

def create_dataloader(file_paths, labels, batch_size, transform=None):
    dataset = AudioDataset(file_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Пример использования
file_paths = [...]  # Список путей к аудио файлам
labels = [...]      # Список меток эмоций
batch_size = 32

dataloader = create_dataloader(file_paths, labels, batch_size)

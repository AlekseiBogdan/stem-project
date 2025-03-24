import os
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm
import gc

# Класс датасета с динамическим определением инструментов
class MoisesDataset(Dataset):
    def __init__(self, root_dir, sr=22050, duration=3, n_fft=510, hop_length=517, save_mix=False):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.save_mix = save_mix
        self.entries = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if not self.entries:
            raise ValueError(f"В директории {root_dir} нет подпапок с записями.")
        
        # Динамически определяем уникальные инструменты из данных
        self.instruments = self._get_unique_instruments()
        print(f"Найдено {len(self.instruments)} уникальных инструментов: {self.instruments}")

        # Фильтруем записи, оставляя только те, у которых есть хотя бы один ненулевой стем
        self.valid_entries = []
        for entry in self.entries:
            entry_dir = os.path.join(self.root_dir, entry)
            available_stems = self._get_available_stems(entry_dir)
            has_valid_stem = False
            for instr in available_stems:
                stem_path = self._find_wav_file(os.path.join(entry_dir, instr))
                if stem_path:
                    y = self._load_audio(stem_path, check_only=True)
                    if y is not None and not np.all(y == 0):
                        has_valid_stem = True
                        break
            if has_valid_stem:
                self.valid_entries.append(entry)
            else:
                print(f"Пропущена запись {entry_dir}: нет ненулевых стемов.")
        self.entries = self.valid_entries
        if not self.entries:
            raise ValueError("Нет записей с ненулевыми стемами. Проверьте датасет.")

    def _get_unique_instruments(self):
        """Собирает уникальный список инструментов из всех файлов data.json."""
        all_instruments = set()
        for entry in self.entries:
            entry_dir = os.path.join(self.root_dir, entry)
            json_path = os.path.join(entry_dir, 'data.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                stems = metadata.get('stems', [])
                for stem in stems:
                    stem_name = stem.get('stemName')
                    if stem_name:
                        all_instruments.add(stem_name)
        return sorted(list(all_instruments))

    def __len__(self):
        return len(self.entries)

    def _find_wav_file(self, instr_dir):
        """Ищет первый .wav файл в папке инструмента."""
        if not os.path.exists(instr_dir):
            return None
        for file in os.listdir(instr_dir):
            if file.lower().endswith('.wav'):
                return os.path.join(instr_dir, file)
        return None

    def _load_audio(self, path, check_only=False):
        """Загружает аудиофайл или возвращает нулевой массив при ошибке."""
        if not path or not os.path.exists(path):
            print(f"Файл не найден: {path}")
            return np.zeros(self.sr * self.duration) if not check_only else None
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True, duration=self.duration if not check_only else 1.0)
            if not check_only and len(y) < self.sr * self.duration:
                y = np.pad(y, (0, self.sr * self.duration - len(y)))
            if np.all(y == 0):
                print(f"Файл {path} содержит только нули.")
            return y
        except Exception as e:
            print(f"Ошибка загрузки файла {path}: {e}")
            return np.zeros(self.sr * self.duration) if not check_only else None

    def _get_available_stems(self, entry_dir):
        """Получает список доступных стемов для записи."""
        json_path = os.path.join(entry_dir, 'data.json')
        if not os.path.exists(json_path):
            print(f"Файл data.json не найден в {entry_dir}")
            return []
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return [stem['stemName'] for stem in metadata.get('stems', []) if stem['stemName'] in self.instruments]

    def __getitem__(self, idx):
        entry_dir = os.path.join(self.root_dir, self.entries[idx])
        available_stems = self._get_available_stems(entry_dir)
        print(f"Запись {entry_dir}: доступные стемы: {available_stems}")

        # Загружаем стемы для всех инструментов, если стема нет — используем нули
        stems = {}
        for instr in self.instruments:
            if instr in available_stems:
                stem_path = self._find_wav_file(os.path.join(entry_dir, instr))
                stems[instr] = self._load_audio(stem_path)
            else:
                stems[instr] = np.zeros(self.sr * self.duration)

        # Создаём микс из всех стемов
        mix = np.zeros(self.sr * self.duration)
        for instr in self.instruments:
            mix += stems[instr]

        # Нормализация микса (добавляем проверку на нули)
        max_abs_mix = np.max(np.abs(mix))
        if max_abs_mix > 0:
            mix = mix / max_abs_mix * 0.95
        else:
            print(f"Микс для записи {entry_dir} полностью нулевой. Пропускаем запись.")
            return None  # Пропускаем запись

        # Сохраняем микс, если указано
        if self.save_mix:
            mix_path = os.path.join(entry_dir, 'mix.wav')
            if not os.path.exists(mix_path):
                sf.write(mix_path, mix, self.sr)

        # Преобразование в спектрограммы
        def to_spectrogram(y):
            S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            # Проверяем на nan в спектрограмме
            if np.any(np.isnan(S)):
                print(f"Обнаружены nan в спектрограмме для записи {entry_dir}")
                return np.zeros((2, (self.n_fft // 2) + 1, int(self.duration * self.sr / self.hop_length) + 1))
            return np.stack([np.abs(S), np.angle(S)], axis=0)

        mix_spec = to_spectrogram(mix)
        targets_spec = np.stack([to_spectrogram(stems[instr]) for instr in self.instruments], axis=0)

        return torch.FloatTensor(mix_spec), torch.FloatTensor(targets_spec)

# Модель UNet с динамическим числом классов
class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=None):
        super(UNet, self).__init__()
        if n_classes is None:
            raise ValueError("n_classes должен быть указан")
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.dconv_down1 = double_conv(n_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 128, 128)
        self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, self.n_classes * 2, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

# Функция для фильтрации None значений в DataLoader
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Функция обучения
def train_model():
    # Конфигурация
    SR = 22050
    DURATION = 5
    N_FFT = 1022
    HOP_LENGTH = 532
    BATCH_SIZE = 8
    ACCUM_STEPS = 2
    EPOCHS = 15
    LEARNING_RATE = 0.0005

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Путь к датасету (замените на свой)
    dataset_path = os.path.join(os.getcwd(), 'data')
    print(f"Путь к датасету: {dataset_path}")

    # Создаём датасет с динамическим списком инструментов
    dataset = MoisesDataset(dataset_path, sr=SR, duration=DURATION, n_fft=N_FFT, hop_length=HOP_LENGTH, save_mix=True)
    n_classes = len(dataset.instruments)  # Количество уникальных инструментов

    # Сохраняем список инструментов в файл
    with open('instruments.txt', 'w') as f:
        f.write('\n'.join(dataset.instruments))
    print("Список инструментов сохранён в instruments.txt")

    # Разделяем на обучающую и валидационную выборки
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Инициализируем модель с динамическим числом классов
    model = UNet(n_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()

    # Цикл обучения
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_batches = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            if batch is None:
                print(f"Пропущен батч {i} в train_loader: все записи нулевые.")
                continue

            mix, targets = batch
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(mix)
                # Преобразуем targets для соответствия выходу модели
                targets = targets.view(targets.size(0), -1, targets.size(3), targets.size(4))
                loss = criterion(outputs, targets) / ACCUM_STEPS

            # Проверяем на nan в train_loss
            if torch.isnan(loss):
                print(f"Обнаружен nan в train_loss на итерации {i}, пропускаем...")
                continue

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUM_STEPS
            train_batches += 1

            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if train_batches > 0:
            train_loss /= train_batches
        else:
            print("Все батчи в train_loader пропущены. Проверьте данные.")
            break

        # Валидация
        model.eval()
        val_loss = 0
        valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    print("Пропущен батч в val_loader: все записи нулевые.")
                    continue

                mix, targets = batch
                mix = mix.to(device)
                targets = targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(mix)
                    targets = targets.view(targets.size(0), -1, targets.size(3), targets.size(4))
                    # Проверяем на nan в outputs и targets
                    if torch.isnan(outputs).any() or torch.isnan(targets).any():
                        print("Обнаружены nan в outputs или targets, пропускаем батч...")
                        continue
                    loss = criterion(outputs, targets)
                    if torch.isnan(loss):
                        print("Обнаружен nan в val_loss, пропускаем батч...")
                        continue
                    val_loss += loss.item()
                    valid_batches += 1

        # Вычисляем средний val_loss только по валидным батчам
        if valid_batches > 0:
            val_loss /= valid_batches
        else:
            val_loss = float('inf')  # Если все батчи содержат nan, устанавливаем val_loss в бесконечность

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Сохраняем модель на основе train_loss, если val_loss — nan или inf
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_dynamic_instr.pth')
            print('Модель сохранена на основе val_loss!')
        elif train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'best_model_dynamic_instr.pth')
            print('Модель сохранена на основе train_loss!')

if __name__ == "__main__":
    train_model()
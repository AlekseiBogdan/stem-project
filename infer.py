import os
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment

# Класс модели (должен совпадать с тем, что использовался при обучении)
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

def load_instruments(instruments_file='instruments.txt'):
    """Загружает список инструментов из файла."""
    if not os.path.exists(instruments_file):
        raise FileNotFoundError(f"Файл {instruments_file} не найден. Убедитесь, что он был создан во время обучения.")
    with open(instruments_file, 'r') as f:
        instruments = [line.strip() for line in f if line.strip()]
    return instruments

def convert_to_wav(input_path, sr=22050):
    """
    Конвертирует аудиофайл в формат .wav с заданной частотой дискретизации.
    Возвращает путь к сконвертированному файлу.
    """
    if input_path.lower().endswith('.wav'):
        # Если файл уже в формате .wav, проверяем частоту дискретизации
        y, file_sr = librosa.load(input_path, sr=None, mono=True)
        if file_sr == sr:
            return input_path
        # Если частота отличается, конвертируем
        temp_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        audio = AudioSegment.from_wav(input_path)
        audio = audio.set_frame_rate(sr).set_channels(1)
        audio.export(temp_path, format='wav')
        return temp_path

    # Для других форматов конвертируем в .wav
    try:
        temp_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(sr).set_channels(1)
        audio.export(temp_path, format='wav')
        return temp_path
    except Exception as e:
        print(f"Ошибка конвертации файла {input_path} в .wav: {e}")
        return None

def separate_audio(input_path, output_dir, model_path='best_model_dynamic_instr.pth', instruments_file='instruments.txt'):
    # Параметры, использованные при обучении
    SR = 22050
    DURATION = 3
    N_FFT = 510
    HOP_LENGTH = 517

    # Загружаем список инструментов
    instruments = load_instruments(instruments_file)
    print(f"Используемые инструменты: {instruments}")

    # Конвертация входного файла в .wav
    wav_path = convert_to_wav(input_path, sr=SR)
    if wav_path is None:
        print("Не удалось конвертировать файл. Прерываем выполнение.")
        return

    # Устройство для инференса
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Инициализация модели
    model = UNet(n_channels=2, n_classes=len(instruments)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Ошибка загрузки модели из {model_path}: {e}")
        return
    model.eval()

    # Загрузка аудиофайла
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as e:
        print(f"Ошибка загрузки аудиофайла {wav_path}: {e}")
        return

    # Разделение аудио на куски
    chunk_size = SR * DURATION
    overlap = chunk_size // 2  # 50% перекрытие
    step_size = chunk_size - overlap
    stems = {instr: [] for instr in instruments}
    window = np.hanning(overlap * 2)

    for i in tqdm(range(0, len(y), step_size), desc='Обработка'):
        start = i
        end = min(i + chunk_size, len(y))
        chunk = y[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        with torch.no_grad():
            spec = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mag_phase = np.stack([np.abs(spec), np.angle(spec)])
            input_tensor = torch.FloatTensor(mag_phase).unsqueeze(0).to(device)

            pred = model(input_tensor)
            pred = pred.squeeze().cpu().numpy()

        for j, instr in enumerate(instruments):
            mag = pred[j * 2]
            phase = pred[j * 2 + 1]
            S = mag * np.exp(1j * phase)
            y_stem = librosa.istft(S, hop_length=HOP_LENGTH)

            # Подгоняем длину y_stem под chunk_size
            if len(y_stem) < chunk_size:
                y_stem = np.pad(y_stem, (0, chunk_size - len(y_stem)))
            elif len(y_stem) > chunk_size:
                y_stem = y_stem[:chunk_size]

            if i == 0:
                stems[instr].append(y_stem)
            else:
                prev_stem = stems[instr][-1]
                overlap_part = prev_stem[-overlap:] * window[:overlap] + y_stem[:overlap] * window[overlap:]
                prev_stem[-overlap:] = overlap_part
                stems[instr].append(y_stem[overlap:])

    # Сохранение результатов
    os.makedirs(output_dir, exist_ok=True)
    for instr in instruments:
        full_audio = np.concatenate(stems[instr])[:len(y)]
        output_path = os.path.join(output_dir, f'{instr}.wav')
        sf.write(output_path, full_audio, SR)
        print(f"Сохранен стем: {output_path}")

    # Удаление временного файла, если он был создан
    if wav_path != input_path and os.path.exists(wav_path):
        os.remove(wav_path)
        print(f"Удален временный файл: {wav_path}")

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Инференс модели для разделения аудио на стемы.")
    parser.add_argument('--input', type=str, required=True, help="Путь к входному аудиофайлу (любой формат)")
    parser.add_argument('--output', type=str, default='output_stems', help="Папка для сохранения стемов (по умолчанию: output_stems)")
    parser.add_argument('--model', type=str, default='best_model_dynamic_instr.pth', help="Путь к файлу модели (по умолчанию: best_model_dynamic_instr.pth)")
    parser.add_argument('--instruments', type=str, default='instruments.txt', help="Путь к файлу со списком инструментов (по умолчанию: instruments.txt)")

    args = parser.parse_args()

    # Проверка существования входного файла
    if not os.path.exists(args.input):
        print(f"Ошибка: Файл {args.input} не существует.")
        return

    print(f"Запуск инференса для файла: {args.input}")
    print(f"Результаты будут сохранены в: {args.output}")
    print(f"Используемая модель: {args.model}")
    print(f"Используемый список инструментов: {args.instruments}")

    # Вызов функции инференса
    separate_audio(args.input, args.output, args.model, args.instruments)

if __name__ == "__main__":
    main()
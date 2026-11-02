from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

CLASSES = ("Present", "Unknown", "Absent")


def plot_class_distribution(patients):
    """Distribuição de rótulos, contando cada paciente uma única vez."""
    counts = patients["murmur"].value_counts().reindex(CLASSES, fill_value=0)
    proportions = 100 * counts / len(patients)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(counts.index, counts.to_numpy())
    for bar, count, proportion in zip(bars, counts, proportions):
        ax.annotate(
            f"{count} ({proportion:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points", ha="center",
        )
    ax.set(
        xlabel="Classificação do sopro",
        ylabel="Número de pacientes",
        title="Distribuição das classes",
    )
    ax.set_ylim(0, max(counts.max() * 1.2, 1))
    fig.tight_layout()
    plt.show()


def audio_headers(paths):
    """Lê metadados WAV sem carregar os sinais completos."""
    rows = []
    for path in paths:
        path = Path(path)
        info = sf.info(str(path))
        rows.append({
            "filename": path.name,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_s": info.duration,
        })
    return pd.DataFrame(rows, columns=["filename", "sample_rate", "channels", "duration_s"])


def plot_duration_distribution(audio_stats):
    """Histograma das durações das gravações de treino."""
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    ax.hist(audio_stats["duration_s"], bins=30, edgecolor="white")
    ax.set(xlabel="Duração (s)", ylabel="Gravações",
           title="Duração das gravações de treino")
    fig.tight_layout()
    plt.show()


def plot_recording_coverage(train_recordings):
    per_patient = train_recordings.groupby("patient_id").size()

    locations = (
        train_recordings["location"]
        .str.split("_").str[0]
        .value_counts()
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))

    per_patient.value_counts().sort_index().plot.bar(
        ax=axes[0], rot=0
    )
    axes[0].set(
        xlabel="Gravações por paciente",
        ylabel="Número de pacientes",
        title="Distribuição de gravações por paciente"
    )

    locations.plot.bar(ax=axes[1], rot=0)
    axes[1].set(
        xlabel="Localização de auscultação",
        ylabel="Número de gravações",
        title="Distribuição por localização"
    )

    fig.tight_layout()
    plt.show()


def build_window_manifest(recordings, window_seconds=4.0, stride_seconds=2.0, sample_rate=4000):
    """Indexa janelas com sobreposição, sem carregar todos os áudios para memória.

    A janela final é alinhada com o fim da gravação, para não descartar o áudio final.
    Ficheiros mais curtos do que uma janela serão preenchidos com zeros ao serem lidos.
    """
    if window_seconds <= 0 or not (0 < stride_seconds <= window_seconds):
        raise ValueError('A duração e o avanço das janelas devem ser positivos; avanço <= duração.')
    target_frames = int(round(window_seconds * sample_rate))
    stride_frames = int(round(stride_seconds * sample_rate))
    if target_frames < 1 or stride_frames < 1:
        raise ValueError('A janela ou o avanço têm menos de uma amostra.')
    required = {'path', 'filename', 'patient_id', 'location', 'murmur'}
    if not required.issubset(recordings.columns):
        raise ValueError(f'Colunas em falta: {sorted(required - set(recordings.columns))}')

    rows = []
    for rec in recordings.itertuples(index=False):
        wav_path = Path(rec.path)
        info = sf.info(str(wav_path))
        if info.samplerate != sample_rate:
            raise ValueError(f'{wav_path.name}: esperados {sample_rate} Hz, obtidos {info.samplerate} Hz.')
        if info.frames <= 0:
            raise ValueError(f'{wav_path.name}: áudio vazio.')
        total = info.frames
        if total <= target_frames:
            starts = [0]
        else:
            starts = list(range(0, total - target_frames + 1, stride_frames))
            final_start = total - target_frames
            if starts[-1] != final_start:
                starts.append(final_start)
        for index, start in enumerate(starts):
            stop = min(start + target_frames, total)
            rows.append({
                'path': wav_path,
                'filename': rec.filename,
                'patient_id': str(rec.patient_id),
                'location': rec.location,
                'murmur': rec.murmur,
                'window_index': index,
                'start_sample': start,
                'stop_sample': stop,
                'total_samples': total,
                'sample_rate': sample_rate,
                'pad_samples': target_frames - (stop - start),
                'start_s': start / sample_rate,
                'stop_s': stop / sample_rate,
            })
    return pd.DataFrame(rows)


def extract_logmel_window(window, *, window_seconds=4.0, sample_rate=4000,
                          n_fft=512, hop_length=128, n_mels=64,
                          fmin=20, fmax=1500):
    """Extrai um log-mel de uma janela indexada, sem usar estatísticas de outros dados.

    A referência dos dB é fixa (1.0), não o máximo de cada janela. A normalização
    estatística deverá ser ajustada apenas aos pacientes de treino de cada experiência.
    """
    if not (0 <= fmin < fmax <= sample_rate / 2):
        raise ValueError('Intervalo de frequências incompatível com a taxa de amostragem.')
    if n_fft <= 0 or hop_length <= 0 or n_mels <= 0:
        raise ValueError('Parâmetros espectrais inválidos.')
    expected_frames = int(round(window_seconds * sample_rate))
    start, stop = int(window['start_sample']), int(window['stop_sample'])
    if start < 0 or stop <= start:
        raise ValueError('Intervalo de leitura inválido.')
    with sf.SoundFile(str(window['path'])) as handle:
        if handle.samplerate != sample_rate:
            raise ValueError(f'Frequência de amostragem diferente de {sample_rate} Hz.')
        if stop > handle.frames:
            raise ValueError('Janela ultrapassa o comprimento real da gravação.')
        handle.seek(start)
        audio = handle.read(frames=stop - start, dtype='float32', always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('Áudio vazio ou com amostras inválidas.')
    audio = audio.mean(axis=1)  # Mantém o processamento seguro para gravações multicanal.
    audio = audio - audio.mean()  # Remoção do componente contínuo (DC), por janela.
    if len(audio) > expected_frames:
        raise ValueError('Janela maior do que a duração definida.')
    audio = np.pad(audio, (0, expected_frames - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio.astype('float32'), sr=sample_rate,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax, power=2.0, center=False,
    )
    features = librosa.power_to_db(mel, ref=1.0, top_db=None).astype('float32')
    if not np.isfinite(features).all():
        raise ValueError('Log-mel com valores inválidos.')
    return features


def plot_logmel_window(logmel, *, sample_rate=4000, hop_length=128, fmin=20, fmax=1500):
    """Ilustração de uma janela transformada, diferente da exploração de áudios inteiros."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = librosa.display.specshow(logmel, sr=sample_rate, hop_length=hop_length,
                                  x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax,
                                  ax=ax)
    ax.set(title='Log-mel-espectrograma de uma janela de treino',
           xlabel='Tempo (s)', ylabel='Frequência (Hz)')
    fig.colorbar(im, ax=ax, label='Energia (dB; referência fixa)')
    fig.tight_layout()
    plt.show()


def fit_logmel_normalizer(train_windows, feature_kwargs=None, progress_every=0):
    """Calcula médias e desvios-padrão por banda mel, APENAS nas janelas de treino.

    Usa somas acumuladas para não manter todos os espectrogramas em memória.
    Ajustar novamente em cada partição de treino, se for usada validação cruzada.
    O progresso pode ser apresentado de progress_every em progress_every janelas.
    """
    if train_windows.empty:
        raise ValueError('Não há janelas de treino.')
    options = feature_kwargs or {}
    total = total_sq = None
    num_frames = 0
    for index, (_, row) in enumerate(train_windows.iterrows(), start=1):
        features = extract_logmel_window(row, **options).astype('float64')
        if total is None:
            total = np.zeros(features.shape[0], dtype='float64')
            total_sq = np.zeros(features.shape[0], dtype='float64')
        total += features.sum(axis=1)
        total_sq += np.square(features).sum(axis=1)
        num_frames += features.shape[1]
        if progress_every and index % progress_every == 0:
            print(f'Normalização: {index}/{len(train_windows)} janelas analisadas')
    mean = total / num_frames
    variance = np.maximum(total_sq / num_frames - mean**2, 1e-12)
    std = np.sqrt(variance)
    return mean.astype('float32'), std.astype('float32')


def normalize_logmel(logmel, mean, std):
    """Aplica estatísticas calculadas apenas no treino às restantes janelas."""
    features = np.asarray(logmel, dtype='float32')
    m = np.asarray(mean, dtype='float32').reshape(-1, 1)
    s = np.asarray(std, dtype='float32').reshape(-1, 1)
    if features.ndim != 2 or m.shape[0] != features.shape[0] or s.shape[0] != features.shape[0]:
        raise ValueError('A normalização deve ter uma estatística por banda mel.')
    if not (np.isfinite(features).all() and np.isfinite(m).all() and np.isfinite(s).all()):
        raise ValueError('A normalização recebeu valores não finitos.')
    if (s <= 0).any():
        raise ValueError('O desvio-padrão deve ser positivo em todas as bandas.')
    return ((features - m) / np.maximum(s, 1e-6)).astype('float32')


def format_model_inputs(logmel):
    """Organiza o mesmo espectrograma para a CNN 2D e para a LSTM."""
    x = np.asarray(logmel, dtype='float32')
    if x.ndim != 2:
        raise ValueError('O log-mel deve ter dimensão (frequências, tempo).')
    return x[..., np.newaxis], x.T.copy()


def prepare_window_inputs(window, mean, std, feature_kwargs=None):
    """Prepara uma janela para ambas as redes, com a normalização ajustada no treino.

    A mesma função deve ser utilizada nos conjuntos de treino, validação e teste.
    Não guarda as janelas processadas em memória.
    """
    features = extract_logmel_window(window, **(feature_kwargs or {}))
    normalized = normalize_logmel(features, mean, std)
    return format_model_inputs(normalized)


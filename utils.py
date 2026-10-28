from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


CLASSES = ("Present", "Unknown", "Absent")
METRIC_WEIGHTS = {"Present": 5.0, "Unknown": 3.0, "Absent": 1.0}


def weighted_accuracy(y_true, y_pred):
    """Weighted accuracy, aplicada a uma previsão por paciente."""
    truth = np.asarray(y_true, dtype=str)
    prediction = np.asarray(y_pred, dtype=str)

    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) == 0 or len(truth) != len(prediction):
        raise ValueError("Os vetores de rótulos devem ser unidimensionais, não vazios e do mesmo tamanho.")
    if not set(truth).issubset(METRIC_WEIGHTS) or not set(prediction).issubset(METRIC_WEIGHTS):
        raise ValueError("Os rótulos devem pertencer às classes Present, Unknown e Absent.")

    weights = np.array([METRIC_WEIGHTS[label] for label in truth], dtype=float)
    return float(np.sum(weights * (truth == prediction)) / np.sum(weights))


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


def choose_examples(train_recordings, n_per_class=1, seed=42):
    """Seleciona gravações de pacientes distintos, uma amostra por classe."""
    rng = np.random.default_rng(seed)
    selected = []
    for label in CLASSES:
        subset = train_recordings.loc[train_recordings["murmur"] == label]
        patient_ids = subset["patient_id"].drop_duplicates().to_numpy()
        if len(patient_ids) < n_per_class:
            raise ValueError(f"Pacientes insuficientes para a classe {label}.")
        chosen_patients = rng.choice(patient_ids, size=n_per_class, replace=False)
        for patient_id in chosen_patients:
            patient_audio = subset.loc[subset["patient_id"] == patient_id]
            selected.append(patient_audio.iloc[int(rng.integers(len(patient_audio)))])
    return pd.DataFrame(selected).reset_index(drop=True)


def read_audio_excerpt(path, seconds=8):
    """Carrega apenas os primeiros segundos de uma gravação."""
    if seconds <= 0:
        raise ValueError("A duração do excerto deve ser positiva.")
    info = sf.info(str(path))
    frames = min(info.frames, int(seconds * info.samplerate))
    signal, sr = sf.read(str(path), frames=frames, dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal, sr


def plot_signal_and_mel(examples, seconds=8, n_fft=512, hop_length=128, n_mels=64):
    """Forma de onda e mel-espectrograma de cada excerto selecionado."""
    if examples.empty:
        raise ValueError("Não foram fornecidas gravações.")

    fig, axes = plt.subplots(
        len(examples), 2, figsize=(11.5, 2.75 * len(examples)),
        squeeze=False, constrained_layout=True,
    )
    color_image = None
    for i, row in enumerate(examples.itertuples(index=False)):
        signal, sr = read_audio_excerpt(row.path, seconds=seconds)
        if len(signal) == 0:
            raise ValueError(f"Gravação sem amostras: {row.filename}")
        t = np.arange(len(signal)) / sr
        axes[i, 0].plot(t, signal, linewidth=0.6)
        axes[i, 0].set(title=f"{row.murmur} — {row.filename}",
                       xlabel="Tempo (s)", ylabel="Amplitude")

        mel = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        color_image = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis="time", y_axis="mel", vmin=-80, vmax=0, ax=axes[i, 1],
        )
        axes[i, 1].set(title="Mel-espectrograma (dB relativos)",
                       xlabel="Tempo (s)", ylabel="Frequência (Hz)")

    fig.colorbar(color_image, ax=axes[:, 1], shrink=0.8, label="dB")
    plt.show()


# --- Preparação das gravações para os modelos ---

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


def format_model_inputs(logmel):
    """Organiza o mesmo espectrograma para a CNN 2D e para a LSTM."""
    x = np.asarray(logmel, dtype='float32')
    if x.ndim != 2:
        raise ValueError('O log-mel deve ter dimensão (frequências, tempo).')
    return x[..., np.newaxis], x.T.copy()


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
    return ((features - m) / np.maximum(s, 1e-6)).astype('float32')


def prepare_window_inputs(window, mean, std, feature_kwargs=None):
    """Prepara uma janela para ambas as redes, com a normalização ajustada no treino.

    A mesma função deve ser utilizada nos conjuntos de treino, validação e teste.
    Não guarda as janelas processadas em memória.
    """
    features = extract_logmel_window(window, **(feature_kwargs or {}))
    normalized = normalize_logmel(features, mean, std)
    return format_model_inputs(normalized)


# --- Funções partilhadas pelo treino e pela avaliação das redes ---

def balanced_window_weights(train_windows):
    """Pondera as janelas, equilibrando classes, pacientes e gravações.

    Cada classe tem o mesmo peso total; dentro de uma classe, cada paciente
    contribui de forma igual e cada gravação do paciente tem o mesmo peso.
    Assim, gravações longas não dominam a função de perda só por criarem
    um maior número de janelas.
    """
    if train_windows.empty:
        raise ValueError('Não há janelas de treino para ponderar.')
    required = {'patient_id', 'filename', 'murmur'}
    if not required.issubset(train_windows.columns):
        raise ValueError(f'Colunas em falta: {sorted(required - set(train_windows.columns))}')
    patient_labels = train_windows[['patient_id', 'murmur']].drop_duplicates()
    if patient_labels['patient_id'].duplicated().any():
        raise ValueError('Foram encontrados diagnósticos inconsistentes para um paciente.')

    n_patients_class = patient_labels.groupby('murmur')['patient_id'].nunique()
    n_recordings_patient = (
        train_windows[['patient_id', 'filename']].drop_duplicates()
        .groupby('patient_id').size()
    )
    n_windows_recording = train_windows.groupby(['patient_id', 'filename'])['filename'].transform('size')

    weights = (
        1.0 / train_windows['murmur'].map(n_patients_class).to_numpy(dtype='float64')
        / train_windows['patient_id'].map(n_recordings_patient).to_numpy(dtype='float64')
        / n_windows_recording.to_numpy(dtype='float64')
    )
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Não foi possível calcular pesos de treino válidos.')
    weights /= weights.mean()  # Peso médio igual a 1 para estabilidade da otimização.
    return weights.astype('float32')


def cache_normalized_logmel(windows, mean, std, cache_path, feature_kwargs=None, progress_every=2000):
    """Guarda log-mel-espectrogramas normalizados em disco, sem os manter em RAM.

    O ficheiro NPY utiliza float16 apenas para reduzir o espaço ocupado em disco;
    os lotes são convertidos para float32 antes de entrarem nas redes. A assinatura
    da cache inclui as janelas, os parâmetros de extração e a normalização, para
    impedir a reutilização acidental de características desatualizadas.
    """
    import hashlib
    import json

    if windows.empty:
        raise ValueError('Não há janelas para processar.')
    options = dict(feature_kwargs or {})
    cache_path = Path(cache_path)
    if cache_path.suffix != '.npy':
        raise ValueError('O ficheiro de cache deve ter a extensão .npy.')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix('.json')
    required = {'path', 'start_sample', 'stop_sample', 'sample_rate'}
    if not required.issubset(windows.columns):
        raise ValueError(f'Colunas em falta: {sorted(required - set(windows.columns))}')

    # O alinhamento entre a ordem das linhas e a ordem das características é essencial.
    keys = windows[['path', 'start_sample', 'stop_sample', 'sample_rate']].copy()
    keys['path'] = keys['path'].map(lambda path: str(Path(path).resolve()))
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes())
    digest.update(np.asarray(mean, dtype='float32').tobytes())
    digest.update(np.asarray(std, dtype='float32').tobytes())
    digest.update(json.dumps(options, sort_keys=True).encode('utf-8'))
    signature = digest.hexdigest()

    if cache_path.is_file() and meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
            if metadata.get('signature') == signature:
                cached = np.load(cache_path, mmap_mode='r', allow_pickle=False)
                if cached.shape[0] == len(windows) and cached.dtype == np.float16:
                    print(f'Características reutilizadas: {cache_path}')
                    return cached
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass  # Reconstrói a cache se estiver corrompida ou desatualizada.

    temporary_path = cache_path.with_name(cache_path.stem + '.tmp.npy')
    stored = None
    try:
        for index, (_, row) in enumerate(windows.iterrows()):
            logmel = extract_logmel_window(row, **options)
            normalized = normalize_logmel(logmel, mean, std)
            if stored is None:
                stored = np.lib.format.open_memmap(
                    str(temporary_path), mode='w+', dtype='float16',
                    shape=(len(windows), *normalized.shape),
                )
            if normalized.shape != stored.shape[1:]:
                raise ValueError('Foram obtidos espectrogramas com dimensões inconsistentes.')
            stored[index] = normalized.astype('float16')
            if progress_every and (index + 1) % progress_every == 0:
                print(f'Características: {index + 1}/{len(windows)} janelas')
        stored.flush()
        del stored
        stored = None
        temporary_path.replace(cache_path)
        meta_path.write_text(json.dumps({
            'signature': signature, 'n_windows': len(windows),
            'feature_options': options, 'dtype': 'float16',
        }, indent=2, ensure_ascii=False), encoding='utf-8')
    finally:
        if stored is not None:
            del stored
        if temporary_path.exists():
            temporary_path.unlink()

    print(f'Características guardadas: {cache_path}')
    return np.load(cache_path, mmap_mode='r', allow_pickle=False)


def aggregate_patient_predictions(windows, probabilities, patients):
    """Obtém uma previsão por paciente através de médias hierárquicas.

    Primeiro calcula a média das probabilidades das janelas de cada gravação;
    depois, a média das gravações do mesmo paciente. Isto evita favorecer
    gravações mais longas e pacientes com mais gravações.
    """
    scores = np.asarray(probabilities, dtype='float64')
    if scores.ndim != 2 or scores.shape != (len(windows), len(CLASSES)):
        raise ValueError('As probabilidades devem ter uma linha por janela e três classes.')
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('As probabilidades contêm valores inválidos.')
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError('As probabilidades de cada janela devem somar 1.')
    required = {'patient_id', 'filename'}
    if not required.issubset(windows.columns) or not {'patient_id', 'murmur'}.issubset(patients.columns):
        raise ValueError('Faltam identificadores de pacientes, gravações ou diagnósticos.')

    probability_columns = [f'prob_{label}' for label in CLASSES]
    indexed = windows[['patient_id', 'filename']].copy().reset_index(drop=True)
    indexed[probability_columns] = scores
    per_recording = indexed.groupby(['patient_id', 'filename'], sort=False)[probability_columns].mean()
    per_patient = per_recording.groupby(level='patient_id', sort=False).mean().reset_index()

    result = patients[['patient_id', 'murmur']].copy().reset_index(drop=True)
    result = result.merge(per_patient, on='patient_id', how='left', validate='one_to_one')
    if result[probability_columns].isna().any().any():
        raise ValueError('Existem pacientes sem gravações ou previsões.')
    result['prediction'] = np.asarray(CLASSES)[
        result[probability_columns].to_numpy().argmax(axis=1)
    ]
    return result

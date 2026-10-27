"""Utilitários de exploração do CirCor DigiScope.

Funções de processamento de áudio, visualização e avaliação reutilizáveis.
As escolhas experimentais e as análises permanecem no notebook principal.
"""

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
    ax.set(xlabel="Diagnóstico", ylabel="Pacientes", title="Distribuição das classes")
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

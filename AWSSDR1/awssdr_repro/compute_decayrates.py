import numpy as np
import librosa
import matplotlib.pyplot as plt


def enframe(data, length, shift):
    total_length = data.shape[0]
    nli = total_length - length + shift
    nDecayRateFrames = int(np.floor(nli / shift))
    output = np.zeros((nDecayRateFrames, length))
    for i in range(nDecayRateFrames):
        output[i, :] = data[i * int(shift):length + i * int(shift)]

    return output


def read_wav(speech_path, fs):
    speech, Fs = librosa.load(speech_path, sr=fs)
    return speech, Fs


def compute_Dr(speech, nFFT, nShift, nFrame, nFbank, DECAYFITFRLEN, fh, fs):
    stft = librosa.core.stft(
        speech,
        n_fft=nFFT,
        hop_length=nShift,
        win_length=nFrame,
        window="hamming",
        center=False,
    )
    lmfe = np.log(librosa.feature.melspectrogram(S=np.abs(stft), n_mels=nFbank))
    xt = np.linspace(1, DECAYFITFRLEN, DECAYFITFRLEN) / fs * nFrame
    xt = xt.reshape(DECAYFITFRLEN, 1)
    nFreqBands, nFrames = lmfe.shape[0], lmfe.shape[1]
    nli = nFrames - DECAYFITFRLEN + fh
    nDecayRateFrames = int(np.floor(nli / fh))
    Q = np.linalg.pinv(np.concatenate((np.ones_like(xt), xt), axis=1))
    decayRates = np.zeros((nFreqBands, nDecayRateFrames))
    for countMelFreqBand in range(nFreqBands):
        decayRates[countMelFreqBand, :] = np.dot(
            Q[1, :], enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
        )

    return decayRates


def show_Dr(speech, nFFT, nShift, nFrame, nFbank, DECAYFITFRLEN, fh, fs):
    stft = librosa.core.stft(
        speech,
        n_fft=nFFT,
        hop_length=nShift,
        win_length=nFrame,
        window="hamming",
        center=False,
    )
    lmfe = np.log(librosa.feature.melspectrogram(S=np.abs(stft), n_mels=nFbank))
    xt = np.linspace(1, DECAYFITFRLEN, DECAYFITFRLEN) / fs * nFrame
    xt = xt.reshape(DECAYFITFRLEN, 1)
    nFreqBands, nFrames = lmfe.shape[0], lmfe.shape[1]
    nli = nFrames - DECAYFITFRLEN + fh
    nDecayRateFrames = int(np.floor(nli / fh))
    Q = np.linalg.pinv(np.concatenate((np.ones_like(xt), xt), axis=1))
    decayRates = np.zeros((nFreqBands, nDecayRateFrames))
    orig = np.zeros((nFreqBands, nDecayRateFrames))
    countMelFreqBand = 11
    temp = enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
    decayRates[countMelFreqBand, :] = np.dot(
        Q[1, :], enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
    )
    orig[countMelFreqBand, :] = np.dot(
        Q[0, :], enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
    )
    x_range = np.linspace(1, DECAYFITFRLEN, DECAYFITFRLEN)
    for it in range(nDecayRateFrames):
        plt.plot(x_range, temp[:, it])
        plt.plot(x_range, xt * decayRates[countMelFreqBand, it] + orig[countMelFreqBand, it])
        x_range = x_range + fh

    plt.show()


def compute_DR_for_figure(speech, nFFT, nShift, nFrame, nFbank, DECAYFITFRLEN, fh, fs):
    stft = librosa.core.stft(
        speech,
        n_fft=nFFT,
        hop_length=nShift,
        win_length=nFrame,
        window="hamming",
        center=False,
    )
    lmfe = np.log(librosa.feature.melspectrogram(S=np.abs(stft), n_mels=nFbank))
    xt = np.linspace(1, DECAYFITFRLEN, DECAYFITFRLEN) / fs * nFrame
    xt = xt.reshape(DECAYFITFRLEN, 1)
    nFreqBands, nFrames = lmfe.shape[0], lmfe.shape[1]
    nli = nFrames - DECAYFITFRLEN + fh
    nDecayRateFrames = int(np.floor(nli / fh))
    Q = np.linalg.pinv(np.concatenate((np.ones_like(xt), xt), axis=1))
    decayRates = np.zeros((nFreqBands, nDecayRateFrames))
    orig = np.zeros((nFreqBands, nDecayRateFrames))
    for countMelFreqBand in range(nFreqBands):
        decayRates[countMelFreqBand, :] = np.dot(
            Q[1, :], enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
        )
        orig[countMelFreqBand, :] = np.dot(
            Q[0, :], enframe(lmfe[countMelFreqBand, :], DECAYFITFRLEN, fh).T
        )

    return decayRates, orig, nDecayRateFrames

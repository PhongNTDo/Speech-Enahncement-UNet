import os.path

import librosa
import numpy as np


def convert_audio_to_frame_stack(audio_data,
                                 frame_length,
                                 frame_shift):
    audio_len = audio_data.shape[0]

    audio_data_list = [audio_data[start: start + frame_length] for start in
                       range(0, audio_len - frame_length + 1, frame_shift)]
    return np.vstack(audio_data_list)


def convert_audio_to_numpy(audio_dir,
                           list_audio_file,
                           sample_rate,
                           frame_length,
                           frame_shift,
                           min_duration):
    list_audio_array = []

    for file in list_audio_file:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration > min_duration:
            list_audio_array.append(convert_audio_to_frame_stack(y,
                                                                 frame_length,
                                                                 frame_shift))
    return np.vstack(list_audio_array)


def mix_voice_noise(voice,
                    noise,
                    nb_samples,
                    frame_length):
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noise_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = noise[id_noise, :] * level_noise
        prod_noise_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noise_voice


def audio_to_magnitude_db_and_phase(n_fft,
                                    frame_shift_fft,
                                    audio):
    stf_audio = librosa.stft(audio, n_fft=n_fft, hop_length=frame_shift_fft)
    stf_audio_magnitude, stf_audio_phase = librosa.magphase(stf_audio)

    stf_audio_magnitude_db = librosa.amplitude_to_db(stf_audio_magnitude, ref=np.max)

    return stf_audio_magnitude_db, stf_audio_phase


def convert_np_audio_to_spectrogram(np_audio,
                                    dim_square_spec,
                                    n_fft,
                                    frame_shift_fft):
    num_of_audio = np_audio.shape[0]

    m_mag_db = np.zeros((num_of_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((num_of_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(num_of_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, frame_shift_fft, np_audio[i]
        )
    return m_mag_db, m_phase

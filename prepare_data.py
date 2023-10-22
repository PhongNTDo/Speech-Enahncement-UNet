import os
import yaml
import numpy as np
import librosa
import soundfile as sf

from handle_data.data_tools import convert_audio_to_numpy
from handle_data.data_tools import mix_voice_noise
from handle_data.data_tools import convert_np_audio_to_spectrogram

with open("config/ConfigTrainingUNet.yaml") as f:
    config = yaml.safe_load(f)
    config_data = config['data']


def main():
    noise_dir = config_data['noise_dir']
    voice_dir = config_data['voice_dir']
    list_noise_files = os.listdir(config_data['noise_dir'])
    list_voice_files = os.listdir(config_data['voice_dir'])

    sample_rate = config_data['sample_rate']
    frame_length = config_data['frame_length']
    frame_shift_noise = config_data['frame_shift_noise']
    frame_shift = config_data['frame_shift']
    min_duration = config_data['min_duration']
    num_of_samples = config_data["num_of_samples"]

    noise = convert_audio_to_numpy(
        noise_dir, list_noise_files, sample_rate, frame_length, frame_shift_noise, min_duration)
    voice = convert_audio_to_numpy(
        voice_dir, list_voice_files, sample_rate, frame_length, frame_shift, min_duration)

    # Mix clean voice vs noise
    prod_voice, prod_noise, prod_noise_voice = mix_voice_noise(
        voice, noise, num_of_samples, frame_length
    )

    n_fft = config_data["n_fft"]
    frame_shift_fft = config_data["frame_shift_fft"]
    path_save_sound = config_data["path_save_sound"]

    if config_data["save_sound"]:
        noisy_voice_long = prod_noise_voice.reshape(1, num_of_samples * frame_length)
        sf.write(path_save_sound + 'noisy_voice_long.wav', noisy_voice_long[0, :], sample_rate)
        voice_long = prod_voice.reshape(1, num_of_samples * frame_length)
        sf.write(path_save_sound + 'voice_long.wav', voice_long[0, :], sample_rate)
        noise_long = prod_noise.reshape(1, num_of_samples * frame_length)
        sf.write(path_save_sound + 'noise_long.wav', noise_long[0, :], sample_rate)

    dim_square_spec = int(n_fft / 2) + 1
    m_amp_db_voice, m_phase_voice = convert_np_audio_to_spectrogram(
        prod_voice, dim_square_spec, n_fft, frame_shift_fft)
    m_amp_db_noise, m_phase_noise = convert_np_audio_to_spectrogram(
        prod_voice, dim_square_spec, n_fft, frame_shift_fft)
    m_amp_db_voice_noise, m_phase_voice_noise = convert_np_audio_to_spectrogram(
        prod_noise_voice, dim_square_spec, n_fft, frame_shift_fft)

    np.save(os.path.join(config_data["path_save_time_serie"], "voice_timeserie"), prod_voice)
    np.save(os.path.join(config_data["path_save_time_serie"], "noise_timeserie"), prod_noise)
    np.save(os.path.join(config_data["path_save_time_serie"], "voice_noise_timeserie"), prod_noise_voice)

    np.save(os.path.join(config_data["path_save_spectrogram"], "voice_amp_db"), m_amp_db_voice)
    np.save(os.path.join(config_data["path_save_spectrogram"], "noise_amp_db"), m_amp_db_noise)
    np.save(os.path.join(config_data["path_save_spectrogram"], "voice_noise_amp_db"), m_amp_db_voice_noise)

    np.save(os.path.join(config_data["path_save_spectrogram"], "voice_pha_db"), m_phase_voice)
    np.save(os.path.join(config_data["path_save_spectrogram"], "noise_pha_db"), m_phase_noise)
    np.save(os.path.join(config_data["path_save_spectrogram"], "voice_noise_pha_db"), m_phase_voice_noise)


if __name__ == "__main__":
    main()

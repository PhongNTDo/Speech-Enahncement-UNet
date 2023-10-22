import os
import yaml
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from handle_data.data_tools import scaled_in, scaled_ou
from sklearn.model_selection import train_test_split
from models.unet import unet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

with open("config/ConfigTrainingUNet.yaml") as f:
    config = yaml.safe_load(f)
    config_data = config["data"]
    config_model = config["model"]


def training():
    X_in = np.load(os.path.join(config_data["path_save_spectrogram"], "voice_noise_amp_db" + ".npy"))
    X_ou = np.load(os.path.join(config_data["path_save_spectrogram"], "voice_amp_db" + ".npy"))
    # Model of noise to predict
    X_ou = X_in - X_ou

    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    # to scale between -1 and 1
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    print(X_in.shape)
    print(X_ou.shape)

    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    X_in = X_in[:, :, :]
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    X_ou = X_ou[:, :, :]
    X_ou = X_ou.reshape(X_ou.shape[0], X_ou.shape[1], X_ou.shape[2], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

    if config_model["training_from_scratch"]:
        model = unet()

    checkpoint = ModelCheckpoint(os.path.join(config_model["weights_path"], config_model['model_name'] + ".h5"),
                                 verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

    model.summary()
    # Training
    history = model.fit(X_train, y_train, epochs=config_model['epochs'], batch_size=config_model['batch_size'],
                        shuffle=True, callbacks=[checkpoint], verbose=1, validation_data=(X_test, y_test))

    # Plot training and validation loss (log scale)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    training()

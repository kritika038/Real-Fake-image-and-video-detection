import numpy as np
from PIL import Image


def fft_score(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((256, 256))

    img_array = np.array(img)

    fft = np.fft.fft2(img_array)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    mean_val = np.mean(magnitude)
    high_freq = magnitude[100:156, 100:156]

    score = np.mean(high_freq) / mean_val

    fake_prob = min(score * 2.5, 1.0)

    return float(fake_prob)
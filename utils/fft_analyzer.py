import numpy as np
import cv2


def fft_score(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return 0.5

    # Resize for consistency
    image = cv2.resize(image, (256, 256))

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Define central low frequency area
    radius = 30
    low_freq = magnitude[
        center_h - radius:center_h + radius,
        center_w - radius:center_w + radius
    ]

    total_energy = np.sum(magnitude)
    low_energy = np.sum(low_freq)

    high_energy_ratio = (total_energy - low_energy) / (total_energy + 1e-8)

    # Normalize more realistically
    # Most natural images: ~0.6–0.85
    # Diffusion images often: slightly higher
    normalized = (high_energy_ratio - 0.6) / 0.4
    normalized = max(0, min(normalized, 1))

    return float(normalized)
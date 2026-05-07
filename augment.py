import cv2
import numpy as np

# severity 1/2/3 기준 파라미터
_LOW_LIGHT_GAMMA = [2.0, 2.8, 3.5]
_BLUR_KERNEL    = [5, 9, 15]
_NOISE_SIGMA    = [5, 15, 25]


def low_light(img: np.ndarray, severity: int = 1) -> np.ndarray:
    gamma = _LOW_LIGHT_GAMMA[severity - 1]
    img_f = img.astype(np.float32) / 255.0
    img_f = np.power(img_f, gamma)
    # Poisson shot noise + Gaussian read noise
    shot = np.random.poisson(img_f * 255.0) / 255.0
    read = np.random.normal(0, 0.01 * severity, img_f.shape).astype(np.float32)
    return np.clip((shot + read) * 255, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, severity: int = 1) -> np.ndarray:
    k = _BLUR_KERNEL[severity - 1]
    # 수평 방향 균일 커널 (단순 handshake 근사)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, severity: int = 1) -> np.ndarray:
    sigma = _NOISE_SIGMA[severity - 1]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def combined(img: np.ndarray, severity: int = 1) -> np.ndarray:
    img = low_light(img, severity)
    img = motion_blur(img, severity)
    return img


AUGMENTATIONS = {
    'normal':         lambda img, s: img.copy(),
    'low_light':      low_light,
    'motion_blur':    motion_blur,
    'gaussian_noise': gaussian_noise,
    'combined':       combined,
}


def augment(img: np.ndarray, condition: str, severity: int = 1) -> np.ndarray:
    assert condition in AUGMENTATIONS, f"Unknown condition: {condition}"
    assert 1 <= severity <= 3, "severity must be 1, 2, or 3"
    return AUGMENTATIONS[condition](img, severity)

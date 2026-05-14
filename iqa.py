import cv2
import numpy as np


def measure_conditions(img: np.ndarray) -> dict:
    """이미지에서 brightness, blur, noise 수치를 추출."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 평균 픽셀 밝기 (0~1), 낮을수록 어두운 이미지
    brightness = float(gray.mean() / 255.0)

    # Laplacian variance, 낮을수록 더 blurry
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Gaussian smoothing 후 차이의 표준편차로 노이즈 추정
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise = float(np.std(gray.astype(np.float32) - smoothed))

    return {'brightness': brightness, 'blur': blur, 'noise': noise}

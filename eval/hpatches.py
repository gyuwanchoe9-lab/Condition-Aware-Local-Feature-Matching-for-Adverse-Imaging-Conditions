import os
import cv2
import numpy as np
from typing import List, Dict


def load_hpatches(data_dir: str) -> List[Dict]:
    """
    HPatches 데이터셋 로드.
    반환: list of dict with keys: img0, img1, H, scene, pair_id
    """
    pairs = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"HPatches not found at {data_dir}. Run: "
                                "wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz")

    for scene in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        img0_path = os.path.join(scene_dir, '1.ppm')
        if not os.path.exists(img0_path):
            continue
        img0 = cv2.imread(img0_path)
        for i in range(2, 7):
            img1_path = os.path.join(scene_dir, f'{i}.ppm')
            h_path    = os.path.join(scene_dir, f'H_1_{i}')
            if not os.path.exists(img1_path) or not os.path.exists(h_path):
                continue
            img1 = cv2.imread(img1_path)
            H    = np.loadtxt(h_path)
            pairs.append({
                'img0':    img0,
                'img1':    img1,
                'H':       H,
                'scene':   scene,
                'pair_id': f'{scene}_1_{i}',
                'type':    'i' if scene.startswith('i_') else 'v',
            })
    return pairs

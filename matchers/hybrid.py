import numpy as np
from iqa import classify
from .learned import SPLightGlueMatcher, LoFTRMatcher


class HybridMatcher:
    """
    IQA 결과에 따라 SP+LightGlue (normal) 또는 LoFTR (degraded)를 선택.
    tau_b, tau_l은 HPatches val split grid search로 결정.
    """

    def __init__(self, device: str = 'cuda', tau_b: float = 0.3, tau_l: float = 50.0):
        self.tau_b = tau_b
        self.tau_l = tau_l
        self.sp_lg  = SPLightGlueMatcher(device=device)
        self.loftr  = LoFTRMatcher(device=device)

    def match(self, img0: np.ndarray, img1: np.ndarray) -> dict:
        cond = classify(img0, tau_b=self.tau_b, tau_l=self.tau_l)
        matcher = self.loftr if cond == 'degraded' else self.sp_lg
        result = matcher.match(img0, img1)
        result['method_used'] = 'loftr' if cond == 'degraded' else 'sp+lg'
        return result


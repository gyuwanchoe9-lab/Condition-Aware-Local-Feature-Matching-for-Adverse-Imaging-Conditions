from .classical import SIFTMatcher, ORBMatcher, CrossCombinationMatcher
from .learned import SuperPointMatcher, SPLightGlueMatcher, LoFTRMatcher
from .hybrid import HybridMatcher

ALL_MATCHERS = {
    'sift':      SIFTMatcher,
    'orb':       ORBMatcher,
    'sp+nn':     SuperPointMatcher,
    'sp+lg':     SPLightGlueMatcher,
    'loftr':     LoFTRMatcher,
    'hybrid':    HybridMatcher,
}

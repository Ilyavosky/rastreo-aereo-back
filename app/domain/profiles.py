import numpy as np


def select_profiles(F: np.ndarray) -> dict[str, int]:
    if len(F) == 0:
        return {}
    low, high = F.min(axis=0), F.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    distance = np.linalg.norm((F - low) / span, axis=1)
    return {
        "best_detection": int(np.argmin(F[:, 0])),
        "lowest_consumption": int(np.argmin(F[:, 2])),
        "balanced": int(np.argmin(distance)),
    }

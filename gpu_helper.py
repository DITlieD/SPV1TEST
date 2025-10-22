"""
GPU Detection Helper for LightGBM
Automatically falls back to CPU if GPU is not available
"""

import lightgbm as lgb

_GPU_AVAILABLE = None

def get_device_type():
    """
    Detects if LightGBM has GPU support enabled.
    Returns 'gpu' if available, 'cpu' otherwise.
    """
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return 'gpu' if _GPU_AVAILABLE else 'cpu'

    try:
        # Try to create a simple GPU dataset
        import numpy as np
        test_data = lgb.Dataset(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        test_params = {'objective': 'binary', 'device': 'gpu', 'verbose': -1}
        lgb.train(test_params, test_data, num_boost_round=1)
        _GPU_AVAILABLE = True
        print("[GPU] [OK] LightGBM GPU support detected and working")
        return 'gpu'
    except Exception as e:
        _GPU_AVAILABLE = False
        print(f"[GPU] [WARN] GPU not available, falling back to CPU. Reason: {e}")
        return 'cpu'

def get_lgbm_params_with_device(base_params):
    """
    Returns LightGBM parameters with appropriate device setting.
    Automatically removes 'device' key if GPU is not available.
    """
    params = base_params.copy()
    device = get_device_type()

    if device == 'cpu' and 'device' in params:
        del params['device']
        print("[GPU] Removed 'device' parameter (using CPU)")
    else:
        params['device'] = device

    return params

def resolve_device_type(preferred_device: str = 'auto') -> str:
    """
    Resolves the device type based on user preference and availability.

    Args:
        preferred_device: 'auto', 'gpu', or 'cpu'.

    Returns:
        'gpu' or 'cpu'.
    """
    if preferred_device == 'cpu':
        print("[Device] User selected 'Force CPU'.")
        return 'cpu'

    # get_device_type() handles auto-detection and caching the result
    available_device = get_device_type() 

    if preferred_device == 'gpu':
        print("[Device] User selected 'Force GPU'.")
        if available_device == 'gpu':
            return 'gpu'
        else:
            print("[Device] WARNING: GPU requested but not available. Falling back to CPU.")
            return 'cpu'

    # Default to auto-detect
    print(f"[Device] Auto-detecting device. Result: {available_device.upper()}")
    return available_device

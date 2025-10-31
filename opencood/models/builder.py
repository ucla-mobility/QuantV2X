"""
Builder functions for models - replacement for mmdet3d's build_model
"""

import sys
import os

# Add mmdet3d to path if it exists
mmdet3d_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'mmdet3d_repo')
if os.path.exists(mmdet3d_path) and mmdet3d_path not in sys.path:
    sys.path.insert(0, mmdet3d_path)

# Try to import from mmcv, fall back to our stub
try:
    from mmcv import Config
    print("[Builder] Using real mmcv.Config")
except ImportError:
    from opencood.utils.mmcv_stub import Config
    print("[Builder] Using stub mmcv.Config")

try:
    from mmdet3d.models import build_model
    print("[Builder] Using real mmdet3d.models.build_model")
except ImportError:
    print("[Builder] WARNING: Could not import mmdet3d.models.build_model")
    print("[Builder] You may need to install mmdet3d or ensure it's in your Python path")

    # Fallback build_model (basic version)
    def build_model(cfg, train_cfg=None, test_cfg=None):
        """Fallback build_model if mmdet3d is not available"""
        raise ImportError(
            "mmdet3d is not available. Please install mmdet3d or ensure "
            "mmdet3d_repo is in your Python path."
        )


def build_detector(detector_cfg_path, train_cfg=None, test_cfg=None):
    """
    Build detector from config file path

    Args:
        detector_cfg_path (str): Path to detector config file (e.g., 'opencood/hypes_yaml/detector_cfgs/fsd_dair.py')
        train_cfg (dict, optional): Training config override
        test_cfg (dict, optional): Testing config override

    Returns:
        nn.Module: Detector model
    """
    # Load config
    detector_cfg = Config.fromfile(detector_cfg_path)

    # Override train/test cfg if provided
    if train_cfg is not None:
        if hasattr(detector_cfg, 'model'):
            detector_cfg.model['train_cfg'] = train_cfg
        else:
            detector_cfg['train_cfg'] = train_cfg

    if test_cfg is not None:
        if hasattr(detector_cfg, 'model'):
            detector_cfg.model['test_cfg'] = test_cfg
        else:
            detector_cfg['test_cfg'] = test_cfg

    # Get model config
    if hasattr(detector_cfg, 'model'):
        model_cfg = detector_cfg.model
    else:
        model_cfg = detector_cfg

    # Build model using mmdet3d's build_model
    model = build_model(
        model_cfg,
        train_cfg=train_cfg or getattr(detector_cfg, 'train_cfg', None),
        test_cfg=test_cfg or getattr(detector_cfg, 'test_cfg', None)
    )

    return model

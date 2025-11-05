"""GPU acceleration configuration for AFS Analysis.

Supports OpenCL (AMD/NVIDIA/Intel GPUs) for accelerated image processing.
Based on AFS_acquisition GPU implementation.
"""

import cv2
from src.utils.logger import Logger

logger = Logger

# GPU Acceleration Configuration
# Supports both AMD (via OpenCL) and NVIDIA GPUs (via OpenCL or CUDA)
GPU_AVAILABLE = False
USE_GPU = False

try:
    # Try OpenCL first (works with AMD, NVIDIA, and Intel GPUs)
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            GPU_AVAILABLE = True
            USE_GPU = True
            logger.info("GPU acceleration available: OpenCL enabled (AMD/NVIDIA/Intel GPU)", "gpu_config")
        else:
            logger.info("OpenCL detected but failed to enable - using CPU", "gpu_config")
    else:
        logger.info("GPU acceleration not available (no OpenCL) - using CPU", "gpu_config")
except Exception as e:
    logger.debug(f"GPU check failed: {e} - using CPU", "gpu_config")
    GPU_AVAILABLE = False
    USE_GPU = False


def get_gpu_info() -> dict:
    """Get information about available GPU acceleration."""
    info = {
        'opencl_available': False,
        'opencl_enabled': False,
        'device_name': 'CPU',
        'acceleration': 'None'
    }
    
    try:
        if cv2.ocl.haveOpenCL():
            info['opencl_available'] = True
            if cv2.ocl.useOpenCL():
                info['opencl_enabled'] = True
                # Try to get device name
                try:
                    device = cv2.ocl.Device.getDefault()
                    info['device_name'] = device.name()
                    info['acceleration'] = 'OpenCL'
                except:
                    info['device_name'] = 'OpenCL Device (Unknown)'
                    info['acceleration'] = 'OpenCL'
    except Exception as e:
        logger.debug(f"Error getting GPU info: {e}")
    
    return info


def disable_gpu():
    """Disable GPU acceleration (useful for debugging or fallback)."""
    global USE_GPU
    USE_GPU = False
    try:
        cv2.ocl.setUseOpenCL(False)
        logger.info("GPU acceleration disabled", "gpu_config")
    except:
        pass


def enable_gpu():
    """Enable GPU acceleration if available."""
    global USE_GPU, GPU_AVAILABLE
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                USE_GPU = True
                GPU_AVAILABLE = True
                logger.info("GPU acceleration enabled", "gpu_config")
                return True
    except:
        pass
    
    USE_GPU = False
    logger.error("GPU acceleration could not be enabled", "gpu_config")
    return False

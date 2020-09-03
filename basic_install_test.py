import torch
import importlib
import warnings

try:
    import deepspeed as ds
    print("deepspeed successfully imported")
except ImportError as err:
    raise err

print(f"torch install path: {torch}")
print(f"deepspeed install path: {ds}")
print(f"torch version: {torch.__version__}")
print(f"deepspeed info: {ds.__version__}, {ds.__git_hash__}, {ds.__git_branch__}")

try:
    importlib.import_module('apex_C')
    print("apex successfully installed")
except ImportError as err:
    raise err

try:
    importlib.import_module('deepspeed.ops.lamb.fused_lamb_cuda')
    print('deepspeed fused lamb kernels successfully installed')
except ImportError as err:
    warnings.warn('deepspeed fused lamb kernels are NOT installed')

try:
    from apex.optimizers import FP16_Optimizer
    print("using old-style apex")
except ImportError:
    print("using new-style apex")

try:
    importlib.import_module('deepspeed.ops.transformer.transformer_cuda')
    print('deepspeed transformer kernels successfully installed')
except ImportError as err:
    warnings.warn("deepspeed transformer kernels are NOT installed")

try:
    importlib.import_module('deepspeed.ops.sparse_attention.cpp_utils')
    print('deepspeed sparse attention successfully installed')
except ImportError:
    warnings.warn('deepspeed sparse attention is NOT installed')

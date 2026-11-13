"""RAFT-Stereo model code (Lipson, Teed, Deng, 3DV 2021), MIT license, see LICENSE.

Vendored from https://github.com/princeton-vl/RAFT-Stereo at 6e93ed2 (core/ only).
Changes: relative imports, opt_einsum replaced by torch.einsum, scipy imported lazily,
torch deprecation fixes (autocast, meshgrid indexing='ij', the old default).
"""
from .raft_stereo import RAFTStereo  # noqa: F401

"""MedNeXt-L (kernel5 / kernel3) + bruniss's SkeletonRecall loss.

Targets the compressed / highly curved surface regions where the production
ResEnc-L surface model loses recall. See issue #191 ("Surface and Fiber
Predictions in Compressed or Highly Curved areas") and the benchmark comment
that motivated this trainer:

  https://github.com/ScrollPrize/villa/issues/191

The corrected PR #975 evaluation uses all 7 held-out S1 cubes from the original
benchmark and the official Kaggle Vesuvius Surface Detection metric
(TopoScore + Surface Dice + VOI): MedNeXt-L + SkeletonRecall scores 0.4397
vs the d058 production model's 0.3996, and a voxel-wise max(d058, MedNeXt)
ensemble scores 0.4437. The downstream ink sanity check on three labeled crops
scores 0.7602 mean AUC for the MedNeXt morph pipeline vs 0.5083 for the d058
morph pipeline. See PR #975 and the supporting writeup for methodology and
caveats.

This trainer keeps everything in ``nnUNetTrainerSkeletonRecall`` (loss,
SkeletonTransform, custom data loaders, train/validation steps) and only
overrides ``build_network_architecture`` to swap the default ResEnc U-Net for
MedNeXt-L with the kernel size and channel/block schedule from
``nnUNetTrainerV2_MedNeXt_L_kernel5`` in the upstream MedNeXt repo
(https://github.com/MIC-DKFZ/MedNeXt). MedNeXt's deep_supervision output list
matches nnUNetv2's expectations (5 levels, full resolution first).

Requirements:
  - The optional ``[mednext]`` extra for the ``MedNeXt`` class. If that package
    is not available in your environment, install MIC-DKFZ/MedNeXt directly.
    Source: https://github.com/MIC-DKFZ/MedNeXt
  - Everything ``nnUNetTrainerSkeletonRecall`` already depends on.

Pretrained weights:
  https://huggingface.co/ciscoriordan/mednext-l-scroll-surface
  (best checkpoint: ``kernel5_skelrec_dataset059_ep33/``)

Usage:
  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5

Memory note:
  MedNeXt-L kernel5 at the default batch_size=2, patch 128^3 fits on a
  40 GB A100 with room to spare. On smaller cards or larger patches, fall
  back to ``nnUNetTrainerSkeletonRecall_MedNeXtL_kernel3``.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecall import (
    nnUNetTrainerSkeletonRecall,
)


def _import_mednext():
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
    except ImportError as exc:
        raise ImportError(
            "MedNeXt trainers require the optional MedNeXt dependency. Install "
            'this nnU-Net package with `pip install -e ".[mednext]"` or install '
            "the MIC-DKFZ/MedNeXt package before selecting a MedNeXt trainer."
        ) from exc
    return MedNeXt


def _unwrap_network(network: nn.Module) -> nn.Module:
    seen_ids = set()
    while id(network) not in seen_ids:
        seen_ids.add(id(network))
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
            continue
        if hasattr(network, "module"):
            network = network.module
            continue
        break
    return network


class _MedNeXtLArchitectureMixin(nnUNetTrainerSkeletonRecall):
    def set_deep_supervision_enabled(self, enabled: bool):
        mod = _unwrap_network(self.network)
        if hasattr(mod, "do_ds"):
            mod.do_ds = enabled
            return
        super().set_deep_supervision_enabled(enabled)

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        kernel_size: int = 5,
    ) -> nn.Module:
        """Match ``nnUNetTrainerV2_MedNeXt_L_kernel5.initialize_network()`` from mednextv1.

        nnUNetv2's deep_supervision_scales for a 5-pool 3D U-Net is 5 levels
        (full, /2, /4, /8, /16). MedNeXt with these settings returns exactly
        that list shape, with ``output[0]`` = full resolution.
        """
        MedNeXt = _import_mednext()
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=kernel_size,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style="outside_block",
        )


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5(_MedNeXtLArchitectureMixin):
    """MedNeXt-L kernel5 architecture + SkeletonRecall loss."""


class nnUNetTrainerSkeletonRecall_MedNeXtL_kernel3(_MedNeXtLArchitectureMixin):
    """Kernel3 fallback variant. Use when kernel5 doesn't fit in VRAM at the
    desired batch_size / patch size. Same channel/block schedule, kernel=3.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _MedNeXtLArchitectureMixin.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            kernel_size=3,
        )

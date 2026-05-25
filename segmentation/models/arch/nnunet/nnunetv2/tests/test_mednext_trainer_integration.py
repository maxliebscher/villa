from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
import unittest
from pathlib import Path


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "nnUNetTrainer"
    / "variants"
    / "loss"
    / "nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5.py"
)


class _BlockMedNeXtImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("nnunet_mednext"):
            raise AssertionError(f"MedNeXt was imported while loading {fullname}")
        return None


class _ModulePatch:
    def __init__(self, modules: dict[str, types.ModuleType], block_mednext: bool = False):
        self.modules = modules
        self.blocker = _BlockMedNeXtImports() if block_mednext else None
        self.previous: dict[str, types.ModuleType | None] = {}
        self.prefixes = tuple(
            {name.split(".", 1)[0] for name in self.modules} | {"torch", "nnunetv2", "nnunet_mednext"}
        )

    def __enter__(self):
        self.previous = {
            name: sys.modules.get(name)
            for name in list(sys.modules)
            if self._is_managed_module(name)
        }
        for name in list(self.previous):
            sys.modules.pop(name, None)
        sys.modules.update(self.modules)
        if self.blocker is not None:
            sys.meta_path.insert(0, self.blocker)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.blocker is not None:
            sys.meta_path.remove(self.blocker)
        for name in list(sys.modules):
            if self._is_managed_module(name):
                sys.modules.pop(name, None)
        for name, module in self.previous.items():
            if module is not None:
                sys.modules[name] = module

    def _is_managed_module(self, name: str) -> bool:
        return any(name == prefix or name.startswith(prefix + ".") for prefix in self.prefixes)


class FakeModule:
    pass


class FakeSkeletonRecallTrainer:
    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.decoder.deep_supervision = enabled


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _base_modules() -> dict[str, types.ModuleType]:
    nn_module = types.SimpleNamespace(Module=FakeModule)
    skeleton_module_name = (
        "nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecall"
    )
    return {
        "torch": _module("torch", nn=nn_module),
        "nnunetv2": _module("nnunetv2"),
        "nnunetv2.training": _module("nnunetv2.training"),
        "nnunetv2.training.nnUNetTrainer": _module("nnunetv2.training.nnUNetTrainer"),
        "nnunetv2.training.nnUNetTrainer.variants": _module(
            "nnunetv2.training.nnUNetTrainer.variants"
        ),
        "nnunetv2.training.nnUNetTrainer.variants.loss": _module(
            "nnunetv2.training.nnUNetTrainer.variants.loss"
        ),
        skeleton_module_name: _module(
            skeleton_module_name,
            nnUNetTrainerSkeletonRecall=FakeSkeletonRecallTrainer,
        ),
    }


def _install_fake_mednext(modules: dict[str, types.ModuleType], created: list[FakeModule]) -> None:
    class FakeMedNeXt(FakeModule):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.do_ds = kwargs["deep_supervision"]
            created.append(self)

    mednext_module_name = "nnunet_mednext.network_architecture.mednextv1.MedNextV1"
    modules.update(
        {
            "nnunet_mednext": _module("nnunet_mednext"),
            "nnunet_mednext.network_architecture": _module(
                "nnunet_mednext.network_architecture"
            ),
            "nnunet_mednext.network_architecture.mednextv1": _module(
                "nnunet_mednext.network_architecture.mednextv1"
            ),
            mednext_module_name: _module(mednext_module_name, MedNeXt=FakeMedNeXt),
        }
    )


def _load_trainer_module(modules: dict[str, types.ModuleType], block_mednext: bool = False):
    with _ModulePatch(modules, block_mednext=block_mednext):
        return _exec_trainer_module()


def _exec_trainer_module():
    spec = importlib.util.spec_from_file_location("mednext_trainer_under_test", TRAINER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MedNeXtTrainerIntegrationTests(unittest.TestCase):
    def test_import_does_not_require_mednext_optional_dependency(self):
        module = _load_trainer_module(_base_modules(), block_mednext=True)

        self.assertTrue(hasattr(module, "nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5"))

    def test_build_network_lazily_instantiates_kernel5_mednext(self):
        modules = _base_modules()
        created: list[FakeModule] = []
        _install_fake_mednext(modules, created)
        with _ModulePatch(modules):
            module = _exec_trainer_module()
            network = module.nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5.build_network_architecture(
                architecture_class_name="ignored",
                arch_init_kwargs={},
                arch_init_kwargs_req_import=(),
                num_input_channels=1,
                num_output_channels=2,
                enable_deep_supervision=False,
            )

        self.assertIs(network, created[0])
        self.assertEqual(network.kwargs["in_channels"], 1)
        self.assertEqual(network.kwargs["n_classes"], 2)
        self.assertEqual(network.kwargs["kernel_size"], 5)
        self.assertFalse(network.kwargs["deep_supervision"])

    def test_deep_supervision_toggle_updates_mednext_do_ds(self):
        module = _load_trainer_module(_base_modules(), block_mednext=True)
        trainer = module.nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5.__new__(
            module.nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5
        )
        trainer.network = types.SimpleNamespace(do_ds=True)

        trainer.set_deep_supervision_enabled(False)

        self.assertFalse(trainer.network.do_ds)

    def test_deep_supervision_toggle_updates_wrapped_mednext_do_ds(self):
        module = _load_trainer_module(_base_modules(), block_mednext=True)
        trainer = module.nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5.__new__(
            module.nnUNetTrainerSkeletonRecall_MedNeXtL_kernel5
        )
        wrapped_network = types.SimpleNamespace(do_ds=True)
        compiled_network = types.SimpleNamespace(_orig_mod=wrapped_network)
        trainer.network = types.SimpleNamespace(module=compiled_network)

        trainer.set_deep_supervision_enabled(False)

        self.assertFalse(wrapped_network.do_ds)


if __name__ == "__main__":
    unittest.main()

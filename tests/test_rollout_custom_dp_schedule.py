from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUM_GPUS = 0


def _install_rollout_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.dtype = type("dtype", (), {})
    torch_mod.Size = tuple
    torch_mod.distributed = SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    ray_mod = types.ModuleType("ray")

    def fake_remote(obj=None, **_kwargs):
        def decorate(cls):
            cls.remote = staticmethod(lambda *args, **kwargs: cls(*args, **kwargs))
            cls.options = staticmethod(
                lambda **_options: SimpleNamespace(remote=lambda *args, **kwargs: cls(*args, **kwargs))
            )
            return cls

        return decorate(obj) if obj is not None else decorate

    ray_mod.remote = fake_remote
    ray_mod.put = lambda value: value
    ray_private = types.ModuleType("ray._private")
    ray_private.services = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
    ray_mod._private = ray_private
    ray_util = types.ModuleType("ray.util")
    ray_sched = types.ModuleType("ray.util.scheduling_strategies")
    ray_sched.PlacementGroupSchedulingStrategy = type("PlacementGroupSchedulingStrategy", (), {})
    ray_util.scheduling_strategies = ray_sched
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray._private", ray_private)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_sched)

    sglang_mod = types.ModuleType("sglang")
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_constants = types.ModuleType("sglang.srt.constants")
    sglang_constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    sglang_constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    sglang_constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang_server_args = types.ModuleType("sglang.srt.server_args")
    sglang_server_args.ServerArgs = type("ServerArgs", (), {})
    sglang_utils = types.ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", sglang_constants)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", sglang_server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)

    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__version__ = "0.0.0"
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    httpx_mod = types.ModuleType("httpx")
    httpx_mod.AsyncClient = type("AsyncClient", (), {})
    httpx_mod.Limits = type("Limits", (), {"__init__": lambda self, *args, **kwargs: None})
    httpx_mod.Timeout = type("Timeout", (), {"__init__": lambda self, *args, **kwargs: None})
    httpx_mod.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "httpx", httpx_mod)

    wandb_mod = types.ModuleType("wandb")
    wandb_mod.run = None
    wandb_mod.finish = lambda *args, **kwargs: None
    wandb_mod.init = lambda *args, **kwargs: None
    wandb_mod.log = lambda *args, **kwargs: None
    wandb_mod.login = lambda *args, **kwargs: None
    wandb_mod.Settings = type("Settings", (), {"__init__": lambda self, *args, **kwargs: None})
    wandb_mod.util = SimpleNamespace(generate_id=lambda: "stub")
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)


def _install_custom_hook_module(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("custom_dp_schedule_fixture")

    class DataSource:
        def __init__(self, args):
            self.args = args

    def rollout_fn(*_args, **_kwargs):
        return None

    def custom_schedule(args, train_parallel_config, total_lengths, *, global_batch_size, group_indices):
        args.custom_schedule_call = {
            "train_parallel_config": train_parallel_config,
            "total_lengths": total_lengths,
            "global_batch_size": global_batch_size,
            "group_indices": group_indices,
        }
        return [[1], [0]], [[[0]], [[0]]], [1], [999]

    module.DataSource = DataSource
    module.rollout_fn = rollout_fn
    module.custom_schedule = custom_schedule
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__


def test_rollout_manager_split_train_data_uses_custom_dp_schedule(monkeypatch):
    _install_rollout_import_stubs(monkeypatch)
    hook_module = _install_custom_hook_module(monkeypatch)
    sys.modules.pop("slime.ray.rollout", None)
    rollout_mod = importlib.import_module("slime.ray.rollout")

    args = SimpleNamespace(
        data_source_path=f"{hook_module}.DataSource",
        rollout_function_path=f"{hook_module}.rollout_fn",
        eval_function_path=f"{hook_module}.rollout_fn",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_dp_schedule_path=f"{hook_module}.custom_schedule",
        debug_train_only=True,
        global_batch_size=256,
    )
    manager = rollout_mod.RolloutManager(args, pg=None)
    manager.set_train_parallel_config(
        {
            "dp_size": 2,
            "cp_size": 1,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
    )

    refs = manager._split_train_data_by_dp(
        {
            "tokens": [[101], [202]],
            "group_ids": [10, 20],
            "response_lengths": [1, 1],
            "rewards": [0.0, 1.0],
            "truncated": [False, False],
            "loss_masks": [[1], [1]],
        }
    )

    assert args.custom_schedule_call["total_lengths"] == [1, 1]
    assert args.custom_schedule_call["global_batch_size"] == 256
    assert args.custom_schedule_call["group_indices"] == [10, 20]
    assert [ref.inner["partition"] for ref in refs] == [[1], [0]]
    assert refs[0].inner["tokens"] == [[202]]
    assert refs[1].inner["tokens"] == [[101]]
    assert refs[0].inner["global_batch_sizes"] == [999]

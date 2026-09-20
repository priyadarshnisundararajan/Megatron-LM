# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantized training tests for the experimental Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from transformer_engine.common.recipe import Format, MXFP8BlockScaling
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import BlockAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
    effective_dtype,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


def _make_mlp(device):
    kwargs = {"params_dtype": torch.bfloat16, "device": device}
    return nn.Sequential(te.Linear(64, 128, **kwargs), nn.GELU(), te.Linear(128, 32, **kwargs))


def _assert_sharded_close(actual, expected):
    """Compare all rows without assuming that packed parameters have equal-sized shards."""
    shards = [None] * dist.get_world_size()
    dist.all_gather_object(shards, actual.to_local().detach().cpu())
    torch.testing.assert_close(torch.cat(shards), expected.detach().cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("parameter_placement", [Shard(0), Replicate()], ids=["zero3", "zero1"])
def test_mxfp8_mlp_training_matches_reference(distributed_setup, parameter_placement):
    """MXFP8 MLP training matches an unsharded model through three Adam updates."""
    device = distributed_setup.device
    if distributed_setup.world_size != 2:
        pytest.skip("MXFP8 grouped DBuffer coverage requires exactly two ranks.")
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")

    torch.manual_seed(2026)
    recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        model = _make_mlp(device)
        reference = _make_mlp(device)

    # The reference owns full FP32 master weights, updated independently of MFSDP.
    main_weights = {}
    for name, parameter in model.named_parameters():
        initial_value = (
            parameter.get_high_precision_init_val()
            if effective_dtype(parameter) == torch.uint8
            else parameter.detach()
        )
        main_weights[name] = nn.Parameter(
            initial_value.to(device=device, dtype=torch.float32).clone()
        )

    @torch.no_grad()
    def sync_reference_weights():
        for name, parameter in reference.named_parameters():
            if effective_dtype(parameter) == torch.uint8:
                parameter.quantize_(main_weights[name])
            else:
                parameter.copy_(main_weights[name])

    sync_reference_weights()
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=device):
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0],
                parameter=[parameter_placement],
                gradient=[Shard(0)],
                optimizer=[Shard(0)],
            ),
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )
    sharded_parameters = dict(model.named_parameters())
    for name, parameter in sharded_parameters.items():
        _assert_sharded_close(parameter, main_weights[name])

    # MLP weights share one quantized group; its BF16 biases form a second group.
    [group] = [g for g in model.parameter_groups if isinstance(g.model_weight, QuantizedDBuffer)]
    assert group.model_weight.rowwise_data.layout.tensor_shapes == ((128, 64), (32, 128))
    assert len(group.fsdp_parameters) == 2
    assert len(model.parameter_groups) == 2
    assert isinstance(group.post_optimizer_model_weight, QuantizedDBuffer)
    assert (group.post_optimizer_model_weight is group.model_weight) is isinstance(
        parameter_placement, Shard
    )
    assert group.main_weight.placements == (BlockAtomic(32),)

    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = FusedAdam(list(main_weights.values()), lr=0.01)
    # Different rank inputs make an incorrect gradient reduction observable.
    torch.manual_seed(1234 + distributed_setup.rank)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
        with te.autocast(recipe=recipe):
            output = model(x)
            reference_output = reference(x)
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        target = torch.randn_like(output)
        (output.float() - target.float()).square().mean().backward()
        (reference_output.float() - target.float()).square().mean().backward()

        for name, parameter in reference.named_parameters():
            # Average full BF16 gradients independently of MFSDP's packed reduce-scatter.
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            _assert_sharded_close(sharded_parameters[name].grad, parameter.grad)
            main_weights[name].grad = parameter.grad.float()

        optimizer.step()
        reference_optimizer.step()
        for name, parameter in sharded_parameters.items():
            _assert_sharded_close(parameter, main_weights[name])
        sync_reference_weights()

    # Also check requantization after the final optimizer update.
    with torch.no_grad(), te.autocast(recipe=recipe):
        torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)

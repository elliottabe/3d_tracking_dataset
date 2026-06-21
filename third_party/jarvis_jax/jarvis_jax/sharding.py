"""Data-parallel sharding utilities for multi-GPU inference.

Establishes the JAX device mesh and batch sharding pattern reused across
sub-projects B/C. See task-11-brief.md for the specification.
"""
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def data_parallel_mesh() -> Mesh:
    """Return a 1-D device mesh over all local JAX devices, axis name 'data'."""
    return Mesh(jax.devices(), axis_names=("data",))


def shard_batch(x, mesh: Mesh):
    """Shard array x along axis 0 across the 'data' axis of *mesh*.

    Args:
        x: JAX array with batch on axis 0.
        mesh: Mesh returned by :func:`data_parallel_mesh`.

    Returns:
        A sharded array with the same dtype/shape as *x* but with axis 0
        distributed across devices.
    """
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def replicate(tree, mesh: Mesh):
    """Replicate every array leaf of *tree* across all devices in *mesh*.

    Needed before the data-parallel jitted train step: the batch is sharded
    across the mesh, so the params/optimizer state must be *replicated* across
    the same devices. Freshly built params are uncommitted (jit replicates them
    automatically), but Orbax restore commits restored arrays to device 0 — so
    after resuming a checkpoint they must be explicitly replicated, or the
    jitted step raises "Received incompatible devices".
    """
    return jax.device_put(tree, NamedSharding(mesh, P()))

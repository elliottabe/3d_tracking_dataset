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

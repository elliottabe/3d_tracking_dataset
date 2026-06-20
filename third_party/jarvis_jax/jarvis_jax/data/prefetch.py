"""Background-thread device prefetcher: overlaps CPU batch IO with GPU compute
by device-putting the next batch while the current step runs."""
import queue
import threading

import jax.numpy as jnp

from jarvis_jax.sharding import shard_batch

_SENTINEL = object()


def prefetch(batch_iter, mesh, depth=2):
    """Yield device-resident, sharded batches from a numpy batch iterator,
    buffering up to `depth` ahead on a background thread."""
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            for batch in batch_iter:
                dev = tuple(shard_batch(jnp.asarray(a), mesh) for a in batch)
                q.put(dev)
        finally:
            q.put(_SENTINEL)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        yield item

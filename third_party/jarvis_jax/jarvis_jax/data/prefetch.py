"""Background-thread device prefetcher: overlaps CPU batch IO with GPU compute
by device-putting the next batch while the current step runs."""
import queue
import threading

import numpy as np

from jarvis_jax.sharding import shard_batch


def prefetch(batch_iter, mesh, depth=2):
    """Yield device-resident, sharded batches from a numpy batch iterator,
    buffering up to `depth` ahead on a background thread. A worker-side
    exception is re-raised on the consumer side (not silently swallowed)."""
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            for batch in batch_iter:
                # np.asarray, NOT jnp.asarray: a jnp array lands on the default
                # device first and device_put then shards it DEVICE-TO-DEVICE,
                # racing the running step's NCCL collectives -- 7 GPUs spin at
                # 100 % waiting for the 8th (hangs at steps 500 and 700 on
                # 2026-09-06; the P3b 7-device hang had the same signature).
                # From host memory every shard is a plain host-to-device copy,
                # which cannot deadlock with a collective.
                dev = tuple(shard_batch(np.asarray(a), mesh) for a in batch)
                q.put(("ok", dev))
        except Exception as e:  # surface to the consumer, don't die silently
            q.put(("err", e))
            return
        q.put(("end", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        tag, payload = q.get()
        if tag == "ok":
            yield payload
        elif tag == "end":
            return
        else:  # "err"
            raise payload

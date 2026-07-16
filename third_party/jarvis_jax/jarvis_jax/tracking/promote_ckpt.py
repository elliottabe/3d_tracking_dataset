"""Promote a CheckpointManager step (e.g. best-val, early-stop) to a StandardCheckpointer 'final'.

Used to deploy an early-stopped V2VNet checkpoint as <run>/final for inference/eval.
"""
import argparse, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    a = ap.parse_args()
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.train.checkpoint import make_manager
    v2v = V2VNet(a.num_joints, a.num_joints, rngs=nnx.Rngs(0))
    gdef, abs_state = nnx.split(v2v)
    mngr = make_manager(a.ckpt_dir)
    restored = mngr.restore(a.step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(abs_state)))
    v2v = nnx.merge(gdef, restored["model"])
    ck = ocp.StandardCheckpointer(); ck.save(a.out, nnx.split(v2v)[1], force=True); ck.wait_until_finished()
    print(f"promoted step {a.step} -> {a.out}")


if __name__ == "__main__":
    main()

import numpy as np, os
from viz.core import io
import stac_mjx.io_dict_to_hdf5 as ioh5

def test_fly_dir_format():
    assert io.fly_dir("/run", 3, 1).endswith("/bouts/bout_00003/fly1")

def test_load_outputs_and_kp2d(tmp_path):
    d = tmp_path/"bouts"/"bout_00001"/"fly0"; d.mkdir(parents=True)
    ioh5.save(str(d/"outputs.h5"), {"kp3d_mm": np.zeros((2,50,3),np.float32),
                                    "mesh_mm": np.zeros((2,300,3),np.float32),
                                    "kp_names": np.array(["a","b"], dtype='S')})
    np.savez(d/"kp2d.npz", kp2d=np.zeros((2,7,50,2),np.float32), conf=np.ones((2,7,50),np.float32))
    o = io.load_outputs(str(tmp_path), 1, 0)
    assert o["kp3d_mm"].shape == (2,50,3) and o["mesh_mm"].shape == (2,300,3)
    kp2d, conf = io.load_kp2d(str(tmp_path), 1, 0)
    assert kp2d.shape == (2,7,50,2) and conf.shape == (2,7,50)

import os, numpy as np, pytest
from viz.views import overlay

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
skip = pytest.mark.skipif(not os.path.isdir(ROOT), reason="courtship run not present")

@skip
def test_overlay_writes_png(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.show="mesh,kp,mask,axis"; a.compare=None; a.bodyalign=False; a.out=str(tmp_path/"o.png")
    assert overlay.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0

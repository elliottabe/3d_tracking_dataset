# tests/test_build_pseudolabel_dataset.py
import numpy as np, json, os
from jarvis_jax.cse.build_pseudolabel_dataset import write_pseudolabel_coco, bbox_from_mask
from jarvis_jax.data.v3 import V3Dataset

def test_roundtrip_loads_in_v3(tmp_path):
    H=W=64; m=np.zeros((H,W),bool); m[20:40,25:45]=True
    kp=np.zeros((50,3),float); kp[:5,0]=30; kp[:5,1]=30; kp[:5,2]=1  # 5 visible
    rec={"file_name":"recA/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
         "rgb":np.zeros((H,W,3),np.uint8),"mask":m,"keypoints":kp,"bbox":bbox_from_mask(m)}
    ann=write_pseudolabel_coco(str(tmp_path),[rec],split="train")
    assert os.path.exists(ann)
    # V3Dataset._load_img reads from <root>/<split>/<file_name> (v3.py __getitem__)
    assert os.path.exists(os.path.join(tmp_path,"train","recA/Cam0/Frame_1.jpg"))
    assert os.path.exists(os.path.join(tmp_path,"sam3_masks/train/recA/Cam0/Frame_1.npz"))
    ds=V3Dataset(str(tmp_path),"train")
    assert len(ds)==1
    img4,kpxy,vis=ds[0]
    assert img4.shape==(448,448,4) and kpxy.shape==(50,2) and vis.shape==(50,)
    assert vis[:5].all() and not vis[5:].any()

def test_bbox_from_mask():
    m=np.zeros((10,10),bool); m[2:6,3:8]=True
    assert bbox_from_mask(m)==[3,2,5,4]

def test_no_visible_keypoints_skipped(tmp_path):
    H=W=64; m=np.zeros((H,W),bool); m[20:40,25:45]=True
    kp=np.zeros((50,3),float)  # all vis==0 -> no visible keypoints
    rec={"file_name":"recB/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
         "rgb":np.zeros((H,W,3),np.uint8),"mask":m,"keypoints":kp,"bbox":bbox_from_mask(m)}
    ann=write_pseudolabel_coco(str(tmp_path),[rec],split="train")
    with open(ann) as f:
        coco=json.load(f)
    assert coco["images"]==[] and coco["annotations"]==[]
    assert not os.path.exists(os.path.join(tmp_path,"train","recB/Cam0/Frame_1.jpg"))

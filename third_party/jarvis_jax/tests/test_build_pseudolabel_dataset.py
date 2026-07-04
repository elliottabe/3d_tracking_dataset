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

def test_two_flies_same_frame_masks_not_overwritten(tmp_path):
    # Two records sharing the same file_name (two flies in one courtship frame,
    # merged into a single write_pseudolabel_coco call). Regression for the bug
    # where the second record's np.savez silently overwrote the first record's
    # npz, leaving V3Dataset._load_mask returning an all-zero mask for fly0.
    H=W=64
    mA=np.zeros((H,W),bool); mA[5:15,5:15]=True        # fly0 mask: top-left corner
    mB=np.zeros((H,W),bool); mB[45:55,45:55]=True       # fly1 mask: bottom-right corner
    kpA=np.zeros((50,3),float); kpA[:5,0]=10; kpA[:5,1]=10; kpA[:5,2]=1
    kpB=np.zeros((50,3),float); kpB[5:10,0]=50; kpB[5:10,1]=50; kpB[5:10,2]=1
    recA={"file_name":"recC/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
          "rgb":np.zeros((H,W,3),np.uint8),"mask":mA,"keypoints":kpA,"bbox":bbox_from_mask(mA)}
    recB={"file_name":"recC/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
          "rgb":np.zeros((H,W,3),np.uint8),"mask":mB,"keypoints":kpB,"bbox":bbox_from_mask(mB)}
    ann=write_pseudolabel_coco(str(tmp_path),[recA,recB],split="train")
    with open(ann) as f:
        coco=json.load(f)
    # one image, two annotations, unique annotation ids
    assert len(coco["images"])==1
    assert len(coco["annotations"])==2
    ann_ids=[a["id"] for a in coco["annotations"]]
    assert len(set(ann_ids))==2

    npz_path=os.path.join(tmp_path,"sam3_masks","train","recC/Cam0/Frame_1.npz")
    z=np.load(npz_path)
    assert z["masks"].shape[0]==2
    assert set(z["ann_ids"].tolist())==set(ann_ids)
    assert z["matched"].all()

    ds=V3Dataset(str(tmp_path),"train")
    assert len(ds)==2
    img0,_,_=ds[0]
    img1,_,_=ds[1]
    mask0_sum=img0[...,3].sum()
    mask1_sum=img1[...,3].sum()
    # neither annotation's mask was overwritten/zeroed, and they differ
    assert mask0_sum>0 and mask1_sum>0
    assert not np.array_equal(img0[...,3], img1[...,3])

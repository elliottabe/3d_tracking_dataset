# tests/test_build_pseudolabel_dataset.py
import numpy as np, json, os
from jarvis_jax.tracking.build_pseudolabel_dataset import (
    write_pseudolabel_coco, bbox_from_mask, PseudoLabelWriter)
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


# ---------------------------------------------------------------------------
# PseudoLabelWriter: incremental (per-bout-streaming) writer (C1)
# ---------------------------------------------------------------------------

def test_incremental_writer_two_calls_loads_in_v3(tmp_path):
    """Two `.add_records()` calls (mirrors streaming one bout at a time) must
    still produce a dataset that loads cleanly via V3Dataset, with globally
    unique image/annotation ids and correct per-annotation masks -- AND the
    two-fly-same-file_name grouping (one image, N annotations, one stacked
    mask npz) must still hold when both flies are passed in ONE add_records
    call, exactly like the original whole-list write_pseudolabel_coco."""
    H=W=64

    # "bout 0": two flies sharing one file_name -> must be grouped into ONE
    # image with TWO annotations when added together in one add_records call.
    mA=np.zeros((H,W),bool); mA[5:15,5:15]=True
    mB=np.zeros((H,W),bool); mB[45:55,45:55]=True
    kpA=np.zeros((50,3),float); kpA[:5,0]=10; kpA[:5,1]=10; kpA[:5,2]=1
    kpB=np.zeros((50,3),float); kpB[5:10,0]=50; kpB[5:10,1]=50; kpB[5:10,2]=1
    recA={"file_name":"recD/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
          "rgb":np.zeros((H,W,3),np.uint8),"mask":mA,"keypoints":kpA,"bbox":bbox_from_mask(mA)}
    recB={"file_name":"recD/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
          "rgb":np.zeros((H,W,3),np.uint8),"mask":mB,"keypoints":kpB,"bbox":bbox_from_mask(mB)}

    # "bout 1": a single fly, a different frame -- added in a SEPARATE call,
    # after bout 0's rgb/mask arrays would have already been dropped.
    mC=np.zeros((H,W),bool); mC[20:30,20:30]=True
    kpC=np.zeros((50,3),float); kpC[10:15,0]=25; kpC[10:15,1]=25; kpC[10:15,2]=1
    recC={"file_name":"recD/Cam0/Frame_2.jpg","img_w":W,"img_h":H,
          "rgb":np.zeros((H,W,3),np.uint8),"mask":mC,"keypoints":kpC,"bbox":bbox_from_mask(mC)}

    writer = PseudoLabelWriter(str(tmp_path), split="train")
    writer.add_records([recA, recB])      # bout 0: BOTH flies together
    writer.add_records([recC])            # bout 1: separate call
    ann_path = writer.finalize()

    with open(ann_path) as f:
        coco = json.load(f)
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 3
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert len(set(ann_ids)) == 3                 # globally unique across calls
    img_ids = [im["id"] for im in coco["images"]]
    assert len(set(img_ids)) == 2

    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    frame1_anns = [a for a in coco["annotations"]
                  if id2file[a["image_id"]] == "recD/Cam0/Frame_1.jpg"]
    frame2_anns = [a for a in coco["annotations"]
                  if id2file[a["image_id"]] == "recD/Cam0/Frame_2.jpg"]
    assert len(frame1_anns) == 2                  # two-fly grouping held
    assert len(frame2_anns) == 1

    npz_path = os.path.join(str(tmp_path), "sam3_masks", "train", "recD/Cam0/Frame_1.npz")
    z = np.load(npz_path)
    assert z["masks"].shape[0] == 2
    assert set(z["ann_ids"].tolist()) == {a["id"] for a in frame1_anns}
    assert z["matched"].all()

    ds = V3Dataset(str(tmp_path), "train")
    assert len(ds) == 3
    mask_sums = []
    for i in range(3):
        img4, kpxy, vis = ds[i]
        assert img4.shape == (448, 448, 4) and kpxy.shape == (50, 2) and vis.shape == (50,)
        mask_sums.append(img4[..., 3].sum())
    # every annotation's mask is present (none clobbered/zeroed by a later write)
    assert all(s > 0 for s in mask_sums)


def test_incremental_writer_matches_one_shot_wrapper(tmp_path):
    """`write_pseudolabel_coco` is a thin wrapper over `PseudoLabelWriter`
    (one add_records + finalize) -- for a single-call dataset the two must
    produce byte-identical COCO json."""
    H=W=64
    m=np.zeros((H,W),bool); m[20:40,25:45]=True
    kp=np.zeros((50,3),float); kp[:5,0]=30; kp[:5,1]=30; kp[:5,2]=1
    rec={"file_name":"recE/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
         "rgb":np.zeros((H,W,3),np.uint8),"mask":m,"keypoints":kp,"bbox":bbox_from_mask(m)}

    root_a, root_b = str(tmp_path/"a"), str(tmp_path/"b")
    ann_a = write_pseudolabel_coco(root_a, [rec], split="train")

    writer = PseudoLabelWriter(root_b, split="train")
    writer.add_records([rec])
    ann_b = writer.finalize()

    with open(ann_a) as f: coco_a = json.load(f)
    with open(ann_b) as f: coco_b = json.load(f)
    assert coco_a == coco_b

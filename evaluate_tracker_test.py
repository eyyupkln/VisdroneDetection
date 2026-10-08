"""
SAHI + BoT-SORT Tracker Evaluation
VisDrone DET-val goruntuleri sequence'lere gore gruplanip
tracker sirali calistirilir, GT ile IoU eslestirmesi yapilir.
Cikti: Precision / Recall / F1 per class + genel + JSON rapor
"""

import cv2
import json
import time
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Boxes

# AYARLAR
VAL_DIR     = Path(r"C:\Users\Monster\source\pycharmProject\VisdroneDetection\datasets\Visdrone\VisDrone2019-DET-val")
MODEL_PATH  = "runs/detect/train-6/weights/best.pt"
OUTPUT_JSON = "tracker_eval_results.json"
CONF_THRESH = 0.45
IOU_THRESH  = 0.50
DEVICE      = "cuda:0"
MAX_FRAMES  = 20   # None=tum val; sayi koy hizli test icin (orn. 50)

CLASS_NAMES = [
    "pedestrian","people","bicycle","car",
    "van","truck","tricycle","awning-tricycle",
    "bus","motor"
]


def compute_iou(b1, b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    u=a1+a2-inter
    return inter/u if u>0 else 0.0


def greedy_match(preds, gts, iou_thr):
    if not preds and not gts: return 0,0,0
    if not preds: return 0,0,len(gts)
    if not gts:   return 0,len(preds),0
    used=set(); tp=0
    for pb in preds:
        best,bj=0.0,-1
        for j,gb in enumerate(gts):
            if j in used: continue
            iou=compute_iou(pb[:4],gb[:4])
            if iou>best: best,bj=iou,j
        if best>=iou_thr: tp+=1; used.add(bj)
    return tp, len(preds)-tp, len(gts)-tp


def load_gt(label_path, w, h):
    boxes=[]
    if not label_path.exists(): return boxes
    with open(label_path) as f:
        for line in f:
            v=line.strip().split()
            if len(v)!=5: continue
            cls,xc,yc,bw,bh=map(float,v)
            x1=(xc-bw/2)*w; y1=(yc-bh/2)*h
            x2=(xc+bw/2)*w; y2=(yc+bh/2)*h
            boxes.append([x1,y1,x2,y2,int(cls)])
    return boxes


def make_tracker():
    a=SimpleNamespace(
        tracker_type="botsort",track_high_thresh=0.5,track_low_thresh=0.1,
        new_track_thresh=0.6,track_buffer=30,match_thresh=0.8,
        gmc_method="sparseOptFlow",proximity_thresh=0.5,appearance_thresh=0.25,
        fallback_id=True,fuse_score=True,with_reid=False,model="auto",device=DEVICE)
    return BOTSORT(args=a)


def prf(tp,fp,fn):
    p=tp/(tp+fp) if (tp+fp)>0 else 0.0
    r=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f=2*p*r/(p+r) if (p+r)>0 else 0.0
    return p,r,f


def main():
    images_dir=VAL_DIR/"images"; labels_dir=VAL_DIR/"labels"

    print("Model yukleniyor...")
    det_model=AutoDetectionModel.from_pretrained(
        model_type="yolov11",model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESH,device=DEVICE)

    all_imgs=sorted(images_dir.glob("*.jpg"))
    if MAX_FRAMES: all_imgs=all_imgs[:MAX_FRAMES]

    seqs=defaultdict(list)
    for p in all_imgs: seqs[p.stem[:7]].append(p)

    print(f"{len(all_imgs)} goruntu | {len(seqs)} sequence | IoU>={IOU_THRESH}\n")

    n=len(CLASS_NAMES)
    TP=[0]*n; FP=[0]*n; FN=[0]*n
    ttl_tp=ttl_fp=ttl_fn=0; ttl_frames=0; t0=time.time()

    for si,(sid,imgs) in enumerate(sorted(seqs.items()),1):
        print(f"[{si:3}/{len(seqs)}] {sid}: {len(imgs)} kare ...",end=" ",flush=True)
        tracker=make_tracker()

        for img_path in sorted(imgs):
            frame=cv2.imread(str(img_path))
            if frame is None: continue
            h,w=frame.shape[:2]
            gt=load_gt(labels_dir/f"{img_path.stem}.txt",w,h)

            res=get_sliced_prediction(
                frame,det_model,
                slice_height=1024,slice_width=1024,
                overlap_height_ratio=0.1,overlap_width_ratio=0.1,verbose=0)

            dets=[[*o.bbox.to_xyxy(),o.score.value,o.category.id]
                  for o in res.object_prediction_list]
            dt=torch.tensor(dets).float() if dets else torch.empty((0,6))
            tracked=tracker.update(Boxes(dt,frame.shape[:2]),frame)
            preds=[[t[0],t[1],t[2],t[3],int(t[6])] for t in tracked]

            for c in range(n):
                pc=[b for b in preds if b[4]==c]
                gc=[b for b in gt   if b[4]==c]
                tp,fp,fn=greedy_match(pc,gc,IOU_THRESH)
                TP[c]+=tp; FP[c]+=fp; FN[c]+=fn
                ttl_tp+=tp; ttl_fp+=fp; ttl_fn+=fn
            ttl_frames+=1
        print("OK")

    elapsed=time.time()-t0

    # RAPOR
    print()
    print("="*72)
    print(f"  SAHI + BoT-SORT TRACKER EVALUATION  (IoU >= {IOU_THRESH})")
    print("="*72)
    print(f"  {'Sinif':<22}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    print("-"*72)

    results={"iou_thresh":IOU_THRESH,"per_class":{}}
    mp=mr=mf=0.0; ac=0

    for c in range(n):
        tp=TP[c]; fp=FP[c]; fn=FN[c]
        p,r,f=prf(tp,fp,fn)
        print(f"  {CLASS_NAMES[c]:<22}{tp:>6}{fp:>6}{fn:>6}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
        results["per_class"][CLASS_NAMES[c]]={"tp":tp,"fp":fp,"fn":fn,"precision":round(p,4),"recall":round(r,4),"f1":round(f,4)}
        if (tp+fp+fn)>0: mp+=p; mr+=r; mf+=f; ac+=1

    print("-"*72)
    micro_p,micro_r,micro_f=prf(ttl_tp,ttl_fp,ttl_fn)
    nc=max(ac,1); mp/=nc; mr/=nc; mf/=nc
    print(f"  {'Micro (Genel)':<22}{ttl_tp:>6}{ttl_fp:>6}{ttl_fn:>6}{micro_p:>8.3f}{micro_r:>8.3f}{micro_f:>8.3f}")
    print(f"  {'Macro (Sinif Ort.)':<22}{'':>6}{'':>6}{'':>6}{mp:>8.3f}{mr:>8.3f}{mf:>8.3f}")
    print("="*72)
    print(f"  Islenen kare : {ttl_frames}")
    print(f"  Sure         : {elapsed/60:.1f} dk")
    results["overall"]={"micro":{"precision":round(micro_p,4),"recall":round(micro_r,4),"f1":round(micro_f,4)},
                        "macro":{"precision":round(mp,4),"recall":round(mr,4),"f1":round(mf,4)},
                        "total_frames":ttl_frames,"elapsed_sec":round(elapsed,1)}
    with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=4,ensure_ascii=False)
    print(f"  Sonuclar     : {OUTPUT_JSON}")
    print("="*72)


if __name__ == "__main__":
    main()

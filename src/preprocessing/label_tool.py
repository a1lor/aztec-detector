import cv2, os, glob, math

classes = ["cacahuatl","teocomitl","tepetla","tlapexohuiloni"]
src_root = "source_images_bmp"
dst_root = "source_labels_yolo"
os.makedirs(dst_root, exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(dst_root, c), exist_ok=True)

state = {"ix":-1,"iy":-1,"drawing":False,"boxes":[],"class_id":0,"img_w":0,"img_h":0,"img_path":"","cls_name":""}

def to_yolo(x1,y1,x2,y2,w,h):
    xc=(x1+x2)/2.0; yc=(y1+y2)/2.0
    bw=abs(x2-x1); bh=abs(y2-y1)
    return xc/w, yc/h, bw/w, bh/h

def mouse(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        state["drawing"]=True; state["ix"]=x; state["iy"]=y
    elif event==cv2.EVENT_MOUSEMOVE and state["drawing"]:
        pass
    elif event==cv2.EVENT_LBUTTONUP:
        state["drawing"]=False
        x1,y1=state["ix"],state["iy"]; x2,y2=x,y
        x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
        if x2-x1>2 and y2-y1>2:
            state["boxes"].append((x1,y1,x2,y2,state["class_id"]))

def load_images():
    items=[]
    for ci,cls in enumerate(classes):
        for ext in ("*.bmp","*.png","*.jpg","*.jpeg"):
            for p in glob.glob(os.path.join(src_root, cls, ext)):
                items.append((cls,p))
    items.sort()
    return items

def save_txt(txt_path, boxes, w, h):
    with open(txt_path,"w") as f:
        for (x1,y1,x2,y2,cid) in boxes:
            xc,yc,bw,bh=to_yolo(x1,y1,x2,y2,w,h)
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

items = load_images()
idx=0
cv2.namedWindow("label"); cv2.setMouseCallback("label", mouse)

while 0<=idx<len(items):
    cls, img_path = items[idx]
    img = cv2.imread(img_path)
    if img is None:
        idx+=1; continue
    state["boxes"]=[]
    state["img_w"]=img.shape[1]; state["img_h"]=img.shape[0]
    state["img_path"]=img_path; state["cls_name"]=cls

    label_dir = os.path.join(dst_root, cls)
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(label_dir, base+".txt")
    if os.path.exists(txt_path):
        with open(txt_path,"r") as f:
            for line in f:
                p=line.strip().split()
                if len(p)==5:
                    cid=int(p[0])
                    xc=float(p[1])*state["img_w"]; yc=float(p[2])*state["img_h"]
                    bw=float(p[3])*state["img_w"]; bh=float(p[4])*state["img_h"]
                    x1=int(xc-bw/2); y1=int(yc-bh/2); x2=int(xc+bw/2); y2=int(yc+bh/2)
                    state["boxes"].append((x1,y1,x2,y2,cid))

    while True:
        vis = img.copy()
        for (x1,y1,x2,y2,cid) in state["boxes"]:
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,f"{classes[cid]}",(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(vis,f"[{idx+1}/{len(items)}] {cls} | class={classes[state['class_id']]} | n:next p:prev s:save r:undo 0-3:switch", (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.imshow("label", vis)
        k=cv2.waitKey(20)&0xFF
        if k==ord('n'):
            break
        if k==ord('p'):
            idx=max(0,idx-2); break
        if k==ord('s'):
            save_txt(txt_path, state["boxes"], state["img_w"], state["img_h"])
        if k==ord('r'):
            if state["boxes"]: state["boxes"].pop()
        if k in [ord('0'),ord('1'),ord('2'),ord('3')]:
            state["class_id"]=int(chr(k))
        if k==27 or k==ord('q'):
            cv2.destroyAllWindows(); quit()
    idx+=1

cv2.destroyAllWindows()
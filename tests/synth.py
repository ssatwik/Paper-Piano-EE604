"""Small deterministic paper/dot scenes for algorithm tests only."""
import cv2
import numpy as np

def scene(shift=(0,0), brightness=1.0, offset=0, cover=None, size=(640,480), homography=None, shadow=False):
    w,h=size; image=np.full((h,w,3),110,np.uint8)
    x,y=90+shift[0],120+shift[1]; pw,ph=460,220
    cv2.rectangle(image,(x,y),(x+pw,y+ph),(245,245,245),-1)
    dots=[]
    for i in range(8):
        p=(x+35+i*55,y+ph//2); dots.append(p); cv2.circle(image,p,7,(15,15,15),-1)
    if cover is not None: cv2.circle(image,dots[cover],16,(190,145,120),-1)
    if shadow:
        ramp=np.linspace(1.0,.68,w,dtype=np.float32)[None,:,None]
        image=np.clip(image.astype(np.float32)*ramp,0,255).astype(np.uint8)
    image=np.clip(image.astype(np.float32)*brightness+offset,0,255).astype(np.uint8)
    if homography is not None:
        homography=np.asarray(homography,np.float32)
        image=cv2.warpPerspective(image,homography,(w,h),borderValue=(110,110,110))
        dots=cv2.perspectiveTransform(np.asarray(dots,np.float32).reshape(-1,1,2),homography).reshape(-1,2)
    return image,np.asarray(dots,np.float32)


from ctypes import resize
from turtle import width
import cv2
import numpy as np
import imutils

frame = cv2.imread('imagenes\img6.jpg')

#red_bajo_1 = np.array([15.1,55,93],np.uint8)
red_bajo_1 = np.array([9,100,200],np.uint8)
red_alto_1 = np.array([16.6,255,255],np.uint8)

# red_bajo_2 = np.array([0,100,200],np.uint8)
# red_alto_2 = np.array([16.6,75,91],np.uint8)

def run():
      
    img_out = imutils.resize(frame, width = 900) #height
    
    img_out_2 = img_out[100:250,250:650]
    #cv2.imshow('frame',img_out)

    frame_HSV = cv2.cvtColor(img_out,cv2.COLOR_BGR2HSV)
    mask_red_1 = cv2.inRange(frame_HSV,red_bajo_1,red_alto_1)
    #mask_red_2 = cv2.inRange(frame_HSV,red_bajo_2,red_alto_2)
    #mask_red = cv2.add(mask_red_1,mask_red_2)
    #mask_red_vis = cv2.bitwise_and(frame,frame, mask = mask_red)

    contornos, _ = cv2.findContours(mask_red_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 4000:
            m = cv2.moments(c)
            if (m["m00"] == 0): m["m00"] = 1
            x = int(m["m10"]/m["m00"])
            y = int(m["m01"]/m["m00"])
            cv2.circle(img_out,(x,y),7,(0,255,0),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_out,'{},{}'.format(x,y),(x+10,y),font,0.75,(0,255,0),1,cv2.LINE_AA)
            #new_contorno = cv2.convexHull(c)
            #cv2.drawContours(img_out, [new_contorno], 0, (255,0,0), 3)
        #cv2.drawContours(img_out_2,[c],-1,(0,255,0),2)
        cv2.imshow('frame recortado',img_out)


    #cv2.imshow('frame recortado',img_out_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  

if __name__ == '__main__':
    run()

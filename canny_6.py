# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import random

font = cv2.FONT_HERSHEY_COMPLEX

imagePath ='/home/bharat/workspace/camera_ws/src/ply_files/7_Color.png'
	# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(imagePath,1)
# print(image.shape) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3,3),np.uint8)
bilateral1 = cv2.bilateralFilter(gray,15,75,75)
cv2.imshow("bila",bilateral1)

ret_thresh,thresh1 = cv2.threshold(bilateral1,0,255,cv2.THRESH_OTSU)
cv2.imshow("otsu",thresh1)

erosion=cv2.erode(thresh1,kernel,iterations=1)
cv2.imshow('abc-ero', erosion)

dialation1 = cv2.dilate(erosion,kernel,iterations=1)
cv2.imshow('abc-dial', dialation1)

# dialation1.convertTo(bilateral1,CV_32F)
canny1= cv2.Canny(dialation1,60,100)
cv2.imshow('abc-can', canny1)


_, contours, hierarchy=cv2.findContours(canny1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

print('Numbers of contours found=' + str(len(contours)))
c = max(contours, key = cv2.contourArea)
# print(c)


approx = cv2.approxPolyDP(c, 0.005* cv2.arcLength(c, True), True)
# n1 = np.asarray(approx)
# n1 = list(dict.fromkeys(n1))

approx = np.unique(approx,axis=0)

cv2.drawContours(image,c,-1,(0,255,0),3)
n = approx.ravel()
print("NNNNNNNNNNNNNNNNNNNNNNNN")
print(n)

coor = []


#centroid
M = cv2.moments(canny1)
cX= int(M["m10"]/M["m00"])
cY = int(M["m01"]/M["m00"])

print(cX)
print(cY)


# print
i=0

for j in n:
    
    if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            coor.append((x,y))
  
            # String containing the co-ordinates.
            string = str(x) + " " + str(y) 
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(string)
            
            cv2.circle(image,(cX,cY),10,(255,255,255),-1)
  
            if(i == 0):
                # text on topmost co-ordinate.
                cv2.putText(image, "Arrow tip", (x, y),
                                font, 1, (255, 0, 0)) 
            else:
                # text on remaining co-ordinates.
                cv2.putText(image, string, (x, y), 
                          font, 0.5, (0, 255, 0)) 
    i = i +1



cv2.imshow('contours',image)

rotated_rect = cv2.minAreaRect(approx)
full_rect = cv2.boundingRect(approx)
# (x_r,y_r,w_r,h_r) = rotated_rect
rr_box = cv2.boxPoints(rotated_rect)
rr_box = np.int0(rr_box)
print("####################################################")
print(rr_box)
# cv2.drawContours(image,[rr_box],0,(0,255,0),2)
# cv2.imshow('rect',image)


print("####################################################")
print(coor)

# print(cv2.pointPolygonTest(rr_box,coor[6],False))

remaining_points = []

for point in coor:
    if(cv2.pointPolygonTest(rr_box,point,False) == 1):
        remaining_points.append(point)
        
print(remaining_points)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

newrem = sorted(remaining_points, key= lambda x: x[0])
print(newrem)

index=0
# rect_subdiv2d = (0, 0, image.shape[1], image.shape[0])
sub_div = cv2.Subdiv2D(full_rect)
# sub_div.insert(remaining_points)
for p in coor:
    sub_div.insert(p)

tri_list = sub_div.getTriangleList()
tri_list = np.array(tri_list, dtype=np.int32)               
print(tri_list)

area_tri =[]

for t in tri_list :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        area = abs(0.5*( (t[0]*(t[3]-t[5])) + (t[2]*(t[5]-t[1])) + (t[4]*(t[1]-t[3])) ))
        area_tri.append(area)
        
        cv2.line(image,pt1,pt2,(0,0,255),2)
        cv2.line(image,pt2,pt3,(0,0,255),2)
        cv2.line(image,pt1,pt3,(0,0,255),2)
        
print(area_tri)
print(tri_list[np.argmin(area_tri)])

notch_points = tri_list[np.argmin(area_tri)]
np1 = (notch_points[0],notch_points[1])
np2 = (notch_points[2],notch_points[3])
np3 = (notch_points[4],notch_points[5])

nps = (np1,np2,np3)

print(nps)

cv2.line(image,np1,np2,(0,0,255),2)
cv2.line(image,np3,np2,(0,0,255),2)
cv2.line(image,np1,np3,(0,0,255),2)
# print(type(area.min()))

# final_coords = tri_list(tri_list.index(min(area)))
# print(final_coords)

dist_frm_cent =[]

#finding apex of notch
for p in nps:
    d = np.sqrt((np.square(cX-p[0]))+np.square(cY-p[1]))
    dist_frm_cent.append(d)

print(dist_frm_cent)

apex = nps[np.argmin(dist_frm_cent)]
print(apex)
    
        
cv2.imshow("new tri",image)
        


cv2.waitKey(0)
cv2.destroyAllWindows()
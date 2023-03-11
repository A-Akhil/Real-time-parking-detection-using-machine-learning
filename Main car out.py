import time
import cv2
import torch
import numpy as np

points = []


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('parking.mp4')
count = 0
area=[(0,515),(475,496),(964,445),(1019,508),(1019,548),(685,598),(210,599),(5,598),(7,510)]
area1=[(0,476),(467,450),(900,406),(830,352),(425,399),(86,399),(14,405)]
area2=[(11,400),(19,377),(366,355),(626,336),(755,319),(806,355),(502,383),(231,397)]
area3=[(11,358),(292,353),(483,343),(711,325),(866,317),(825,305),(507,320),(296,334),(155,334),(27,343)]
area4=[(13,329),(40,269),(215,261),(387,253),(571,288),(276,315),(36,326)]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in d:
            results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            results=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            results=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            results=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            results=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            list.append([cx])
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
    a=111-(len(list))
    cv2.putText(frame, "Remaining Slot"+str(a), (74,32), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback("FRAME", POINTS)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
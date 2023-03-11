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

cap = cv2.VideoCapture('parking1.mp4')
count = 0
area=[(4,500),(22,419),(433,404),(896,401),(975,470),(423,514)]
area1=[(1,404),(75,355),(440,337),(815,324),(850,362),(406,389),(172,400)]
area2=[(779,287),(743,269),(689,272),(664,251),(619,260),(609,206),(411,225),(379,309)]
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
           # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            list.append([cx])
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    a=50-(len(list)+27)
    cv2.putText(frame, "Remaining Slot"+str(a), (74,32), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback("FRAME", POINTS)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
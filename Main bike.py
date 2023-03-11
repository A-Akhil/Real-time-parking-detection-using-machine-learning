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

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

cap = cv2.VideoCapture('parking2.mp4')
count = 0
area=[(122,583),(175,483),(244,367),(290,364),(298,338),(358,337),(336,299),(336,265),(244,253),(196,252),(105,344),(93,469),(38,579)]
area1=[(361,304),(358,263),(348,234),(481,205),(574,208),(639,292),(376,309)]
area2=[(805,295),(848,291),(1019,508),(1019,569),(989,571),(873,401)]
area3=[(575,360),(575,331),(352,341),(347,359)]
area4=[(861,472),(924,574),(692,578),(661,488)]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame, size=640)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        name = row['name']
        conf = row['confidence']
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'bikes' in name:
            results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            results = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            results = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            results = cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False)
            results = cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            list.append([cx])

    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
    a=110-(len(list))
    cv2.putText(frame, "Remaining Slot "+str(a), (74,32), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback("FRAME", POINTS)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

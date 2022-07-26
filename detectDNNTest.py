# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
import cv2
import time
import imutils
import argparse
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream
from personObject import Person
import dlib




prototxt_path = '.\Caffe\SSD_MobileNet_prototxt.txt'
model_path = '.\Caffe\SSD_MobileNet.caffemodel'
MODEL_CONFIDENCE = 0.75

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
font_thickness = 1

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


h_bins = 50
s_bins = 80
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]
compare_method = cv2.HISTCMP_KL_DIV

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Tracking
person = None
count = 0

tracker = None
label = ""


prev_frame_time = 0
new_frame_time = 0

acumFPS = 0
 


#Loop Video Stream
while True:

    #Resize Frame to 400 pixels
       # Read a new frame
    frame = vs.read()
    new_frame_time = time.time()
    FPS = 1/(new_frame_time-prev_frame_time)
    


    # ret, frame = cap.read()
    
    frame = imutils.resize(frame, width=400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]


    if(count == 0  and tracker is None and person is None):
        #Converting Frame to Blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        #Passing Blob through network to detect and predict
        nn.setInput(blob)
        detections = nn.forward()

  
        #Loop over the detections
        for i in np.arange(0, detections.shape[2]):
        

	    #Extracting the confidence of predictions
            confidence = detections[0, 0, i, 2]

            #Filtering out weak predictions
            if confidence > MODEL_CONFIDENCE:
            
                #Extracting the index of the labels from the detection
                #Computing the (x,y) - coordinates of the bounding box        
                idx = int(detections[0, 0, i, 1])
                
                if(idx == 15):


                    #Extracting bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    #Drawing the prediction and bounding box
                    label = "{}: {:.2f}%".format('person', confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (232, 229, 59 ), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 229, 59 ), 1)

                    roi = frame[startY:startY+endY, startX:startX + endX]
                    person = Person(count, startX, startY, endX, endY, roi, confidence)
                    
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    
                    count += 1
            else:
                pass
                    
    else:
        
        tracker.update(rgb)
        pos = tracker.get_position()
        person.x = int(pos.left()) 
        person.y = int(pos.top())
        person.w = int(pos.right())
        person.h = int(pos.bottom())
        personImg = frame[person.y:person.y+person.h, person.x :person.x  + person.w]

        
        (pw1, ph1, d1) = person.img.shape
        
        (pw, ph, d) = personImg.shape

        

        if pw > 0 and ph >0  and pw1 > 0 and ph1 >0:
            

            personImg = cv2.resize(personImg,(ph1, pw1),interpolation= cv2.INTER_LINEAR)

            hsv_base = cv2.cvtColor(person.img, cv2.COLOR_BGR2HSV)
            hsv_tracking = cv2.cvtColor(personImg, cv2.COLOR_BGR2HSV)
            
            hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
            hist_test = cv2.calcHist([hsv_tracking], channels, None, histSize, ranges, accumulate=False)
            
            cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            compare = cv2.compareHist(hist_base, hist_test, compare_method)
            	
            # gray_base = cv2.cvtColor(person.img, cv2.COLOR_BGR2GRAY)
            # gray_tracking = cv2.cvtColor(personImg, cv2.COLOR_BGR2GRAY)
            
            # sift = cv2.SIFT_create()
            # # check keypoints and descriptions of images
            # kp_base, desc_base = sift.detectAndCompute(gray_base,None)
            # kp_tracking, desc_tracking = sift.detectAndCompute(gray_tracking,None)
            # # index_params = dict(algorithm=0, trees=5)
            # # search_params = dict()
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            # # search_params = dict(checks=50)   # or pass empty dictionary
            # search_params = dict()   # or pass empty dictionary
            # flann = cv2.FlannBasedMatcher(index_params, search_params)

           


            # if desc_base is not None and len(desc_base) > 2 and desc_tracking is not None and len(desc_tracking) > 2:
            
                
            #     desc_base= desc_base.astype(np.float32, copy=False)
            #     desc_tracking= desc_tracking.astype(np.float32, copy=False)
              

            #     matches = flann.knnMatch(desc_base, desc_tracking, k=2)
                

            #     good_points = []
            #     ratio = 0.3
            #     for m, n in matches:
            #         if m.distance < ratio*n.distance:
            #             good_points.append(m)


            #     result = cv2.drawMatchesKnn(gray_base,kp_base,gray_tracking,kp_tracking,matches,None)
                

            #     print(f'good points {len(good_points)}  compare histogram {compare}')
            #     print(compare > 10 and len(good_points) > 25)

            #     cv2.imshow("Correlation", result)
                
            print(compare)
            if compare > 10 :
                print(f'trackin')
                acumFPS = 0
            
                person.img = personImg
        

                cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
                cv2.imshow("ROI", person.img)

                cv2.namedWindow('TRACKER', cv2.WINDOW_NORMAL)
                cv2.imshow("TRACKER", personImg)

                #Drawing the prediction and bounding box
                acc = "{:.2f}".format(person.accuracy)
                label = f"{acc} -> tracking person {person.id}"
                cv2.rectangle(frame, (person.x, person.y), (person.w, person.h), (232, 229, 59 ), 2)
                y = person.y - 15 if person.y - 15 > 15 else person.y + 15
                cv2.putText(frame, label, (person.x, y),font, fontScale, (232, 229, 59 ), font_thickness)
            else:
                
                acumFPS += FPS
                if acumFPS > 100:
                    print('Tracking Lost by FPS NO GOOD poiNTS')
                    count = 0
                    person = None
                    tracker = None
                    label = f'Tracking Lost FPS acum {acumFPS}'
                    acumFPS = 0
                    cv2.putText(frame, label, (10, 40),font, fontScale, (232, 229, 59 ), font_thickness)

        else:
            acumFPS += FPS
            print(f'lost track  {acumFPS}')
            if acumFPS > 150:
                print('Tracking Lost by FPS')
                count = 0
                person = None
                tracker = None
                label = f'Tracking Lost FPS acum {acumFPS}'
                acumFPS = 0
                cv2.putText(frame, label, (10, 40),font, fontScale, (232, 229, 59 ), font_thickness)


    prev_frame_time = new_frame_time
    cv2.putText(frame, str(int(FPS)), (10, 5), font, fontScale, (100, 255, 0), font_thickness, cv2.LINE_AA)   



    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    fps.update()

 

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

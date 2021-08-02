# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from matplotlib import pyplot as plt 
from skimage import data, filters
import glob
f=False
far=.5
imx=[]
imy=[]
imw=[]
imh=[]
con=[]
t1=str(200)
# construct the argument parse and parse the arguments
def unsharp_mask(image, kernel_size=(3,3), sigma=2.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def YOLO_V3(imagename):
    #YOLO Parameters
    confidence_thr = 0.30
    threshold = 0.3

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    #print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(imagename)
    #Applying Erosion 
    image = unsharp_mask(image)
    #Applying Dilation
    # img_dilation = cv2.dilate(image, kernel, iterations=1)

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID != 0:
                #print(classID)
                #print(confidence)
                if not ((classID>=0 and classID<=5) or (classID==7)):
                    classID = 0
                    confidence = 0
                    continue
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_thr:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence,threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        f=True
        if f:
            con.append(confidences[0])
            f=False
        #far=confidences[0]
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            imx.append(x)
            imy.append(y)
            imw.append(w)
            imh.append(h)
            #print("hoga mara")
            # draw a bounding box rectangle and label on the image
            #print("ClassID"," ",LABELS[classIDs[i]])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    # show the output image
    #print(boxes[0])
    #if imx and con[0]>=.4:
        #cv2.imwrite("C:/Users/mi_em/Desktop/Project1/exp/"+t1+".jpg",image)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

#YOLO_V3("C:/Users/mi_em/Desktop/Project/testing_doc/17550hola.jpg")
#YOLO_V3("C:/Users/mi_em/Desktop/Project/testing_doc/1.jpg")
#"""
# detect stationary vehicles from avg_200
#"""
for al in range(1,101): # Video range
    try:    
        #print(al)
        con=[]
        imx=[]
        imy=[]
        imw=[]
        imh=[]
        imx_1=[]
        imy_1=[]
        imw_1=[]
        imh_1=[]
        #con=[]
        frame_no=450
        imgList = glob.glob("E:/Project/Process_frames/avg_450/test-data/"+str(al)+"/*.jpg")
        for i in range(len(imgList)-1):
            t1=str((i+1)*450)
            name="E:/Project/Process_frames/avg_450/test-data/"+str(al)+"/"+t1+".jpg"
            #print(name)
            #imm=cv2.imread(name)
            #cv2.imshow("im",imm)
            YOLO_V3(name)
            l=len(imx)-1
            if imx and con[0]>=.4:
                print("Video ",al," frame ",t1)
                frame_no=(i+1)*450
                #imx_1=imx[l]
                #imy_1=imy[l]
                #imw_1=imw[l]
                #imh_1=imh[l]
                #print(far)
                #f=False
                break
            else:
                imx=[]
                imy=[]
                imw=[]
                imh=[]
                con=[]
        #imx=[]
        #imy=[]
        #imw=[]
        #imh=[]
        con=[]
        for i in range(len(imx)):
            imx_1.append(imx[i])
            imy_1.append(imy[i])
            imw_1.append(imw[i])
            imh_1.append(imh[i])
            #print("object no. ",i,": ","x: ",imx_1[i]," y: ",imy_1[i]," w: ",imw_1[i]," h: ",imh_1[i])
        #print(frame_no,": ",imx_1," ",imy_1," ",imw_1," ",imh_1)
        final_frame=0
        final_frame=frame_no
        #print("frame_no: ",frame_no)
        # # detect stationary vehicles from avg_30
        if imx:
            #print("ok")
            #for i in range((frame_no-(frame_no%30)),0,-30):
            for i in range((frame_no-(frame_no%20)),21,-20):
                #t2=
                #if i * 20 > frame_no:
                    #break
                imx=[]
                imy=[]
                imw=[]
                imh=[]
                t1=str(i)
                name="E:/Project/Process_frames/avg_20/test-data/"+str(al)+"/"+t1+".jpg"
               # print(name)
                YOLO_V3(name)
                cnt=0
                for jj in range(len(imx_1)):
                    for j in range(len(imx)):
                         
                        if ((imx_1[jj] >= imx[j] and imx_1[jj]-3 <= imx[j]) or (imx_1[jj] <= imx[j] and imx_1[jj]+3 >= imx[j])) and ((imy_1[jj] >= imy[j] and imy_1[jj]-3 <= imy[j]) or (imy_1[jj] <= imy[j] and imy_1[jj]+3 >= imy[j])):
                        #and (imw_1 <= imw[j]+5 and imw_1>=imw[j]-5) and(imh_1 <= imh[j]+5 and imh_1>=imh[j]-5)):
                            #print(t1,": ",imx[j]," ",imy[j]," ",imw[j]," ",imh[j]) 
                            #print()
                            final_frame=i
                            cnt=1
                            break
                    #else:
                       # print(t1,":- ",imx[j]," ",imy[j]," ",imw[j]," ",imh[j])
                    if cnt == 0:
                        break
                #if cnt == 0:
                    #break
            #name1="C:/Users/mi_em/Desktop/Project/Process_frames/avg_20/test-data/"+str(al)+"/"+str(final_frame)+".jpg"
            #img_load=cv2.imread(name1)
            print("Final Frame: ",final_frame)
            res=str(float(final_frame/30.0))
            os.makedirs("E:/Project/result/"+str(al))
            try:
                name1="E:/Project/Process_frames/avg_20/test-data/"+str(al)+"/"+str(final_frame)+".jpg"
                im=cv2.imread(name1)
                cv2.imwrite("E:/Project/result/"+str(al)+"/"+res+".jpg",im)
            except:
                name1="E:/Project/Process_frames/avg_450/test-data/"+str(al)+"/"+str(final_frame)+".jpg"
                im=cv2.imread(name1)
                cv2.imwrite("E:/Project/result/"+str(al)+"/"+res+".jpg",im)
    except:
        print("Problem")
#"""

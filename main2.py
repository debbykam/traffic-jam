import urllib.request as ur
import cv2
import numpy as np
import requests


#CAP = cv2.VideoCapture("video.mp4")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
imageLink = "https://public.highwaystrafficcameras.co.uk/cctvpublicaccess/images/52660"
Frame = None
Frame_line = None
Frame_rest = None
Points_ori = []
counting_line = []
RminX = None
RminY = None
numbers_cars = 0
move_list =[]

def ROI(event,x,y,flag,args):
    global Frame,Points_ori,Frame_rest
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(Frame,(x,y),3,(0,0,255),-1)
        Points_ori.append((x, y))
        if len(Points_ori) >=2:
            cv2.line(Frame,(Points_ori[-1]),(Points_ori[-2]),(0,255,0),2)
    if event == cv2.EVENT_RBUTTONDOWN:
        Frame = Frame_rest.copy()
        Points_ori = []
    cv2.imshow("select_ROI", Frame)

def Counting_line(event, x, y, flag, args):
        global Frame_line, counting_line
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(Frame_line, (x, y), 3, (0, 0, 255), -1)
            counting_line.append((x, y))
            if len(counting_line) >= 2:
                cv2.line(Frame_line, (counting_line[-1]), (counting_line[-2]), (0, 0, 255), 2)
        if event == cv2.EVENT_RBUTTONDOWN:
            counting_line =[]
        cv2.imshow("select coutning line", Frame_line)



def model_Prediction(net,img):

    car_list =[]
    #imgv2 = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()

    outs = net.forward(output_layers_name)

 # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.25:
                    # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN

        if len(indexes) > 0:
            for i in indexes.flatten():
                if i in [2, 3, 5, 7]:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,251),thickness=1)
                    car_list.append([x,y,w,h])


    return img, len(car_list)


def roi(img):
    minX = img.shape[1]
    maxX = -1
    minY = img.shape[0]
    maxY = -1
    for point in Points_ori:
        x = point[0]
        y = point[1]

        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY =y

    cropImage = np.zeros_like(img)
    for y in range(0,img.shape[0]):
        for x in range(img.shape[1]):

            if x <minX or x > maxX or y < minY or y > maxY:
                continue

            if cv2.pointPolygonTest(np.asarray(Points_ori),(x,y),False) >= 0:
                cropImage[y,x,0] = img[y,x,0]
                cropImage[y,x,1] = img[y,x,1]
                cropImage[y, x, 2] = img[y, x, 2]
    finalimg = cropImage[minY:maxY,minX:maxX]

    return finalimg





def DrawContours(mask,img):
    global move_list
    move_list.clear()

    keypoints,_ = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in keypoints:
        if cv2.contourArea(cnt) > 300:
            x,y,w,h = cv2.boundingRect(cnt)
            move_list.append([x,y,w,h])

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = ur.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def main():
    global Frame_line,move_list,numbers_cars,imageLink
    # tracker = EuclideanDistTracker()
    # subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    sumCars = 0

    while True:

        filename = url_to_image(imageLink)
        # download image using GET
        #rawImage = requests.get(imageLink, stream=True)
        # save the image received into the file
        # with open(filename, 'wb') as fd:
        #     for chunk in rawImage.iter_content(chunk_size=1024):
        #         fd.write(chunk)
        # Frame= rawImage
        Frame, number_of_cars = model_Prediction(net,filename)
        sumCars += number_of_cars

        if sumCars > numbers_cars:
            numbers_cars = sumCars
            print(f" sum cars {numbers_cars}")

        # Frame_line = roi(Frame)
        cv2.imshow("select coutning line", Frame)
        # cv2.setMouseCallback("select coutning line", Counting_line)
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.waitKey(5000)
    cv2.destroyAllWindows()


def findborders():
    global RminX, RminY
    minX = Points_ori[0][0]
    minY = Points_ori[0][1]

    for point in Points_ori:
        if point[0] < minX:
            minX = point[0]
        if point[1] < minY:
            minY = point[1]

    RminX = minX
    RminY = minY

if __name__ == '__main__':

   # filename = url_to_image(imageLink)
   # cv2.imshow("select_ROI", filename)
   # Frame_rest = filename.copy()
   # cv2.setMouseCallback("select_ROI", ROI)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   # findborders()
   main()



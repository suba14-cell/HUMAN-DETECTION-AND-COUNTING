from flask import Flask,render_template,request,Response
import cv2
import numpy as np
from persondetection import DetectorAPI
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/startpage')
def startpg():
    return render_template('startpage.html')

@app.route('/video')
def vidpg():
    return render_template('vidhtml.html')                                                  

@app.route('/image')
def imgpg():
    return render_template('imghtml.html')

ALLOWED_EXTENSIONS=['mp4']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/prediction",methods=["POST"])
def prediction():
    imgpath=request.files['image']
    imgpath.save("img.jpg")
    net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    person_class_id = classes.index('person')

    # Set up object detection parameters
    confidence_threshold = 0.35
    nms_threshold = 0.5

# Load image and pass it through object detection model
    img = cv2.imread('img.jpg')
    image = cv2.resize(img, (800, 500))
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

# Extract detected objects labeled as "person"
    persons = []
    confidences = []
    for output in outputs:
       for detection in output:
           scores = detection[5:]
           class_id = np.argmax(scores)
           confidence = scores[class_id]
           if class_id == person_class_id and confidence > confidence_threshold:
               box = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
               center_x, center_y, width, height = box.astype('int')
               x = int(center_x - width/2)
               y = int(center_y - height/2)
               persons.append([x, y, int(width), int(height)])
               confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(persons, confidences, confidence_threshold, nms_threshold)
    persons = [persons[i] for i in indices]           

# Count number of persons
    count = len(persons)

# Draw bounding boxes and count on image
    for person in persons:
        x, y, width, height = person
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2)
    cv2.putText(image, f'Person count: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imencode('.jpg',img)
    #yield(b'--image\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

    #return render_template("prediction.html",data=prediction())
    return render_template("prediction.html",data=prediction())    

@app.route("/cameradetection")
def camera():
     net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
     classes = []
     with open('coco.names', 'r') as f:
         classes = [line.strip() for line in f.readlines()]
     person_class_id = classes.index('person')

# Set up object detection parameters
     confidence_threshold = 0.5
     nms_threshold = 0.4

     def process_frame(frame):
    # Pass frame through object detection model
        blob = cv2.dnn.blobFromImage(frame, 1/255, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Extract detected objects labeled as "person"
        persons = []
        for output in outputs:
            for detection in output:
               scores = detection[5:]
               class_id = np.argmax(scores)
               confidence = scores[class_id]
               if class_id == person_class_id and confidence > confidence_threshold:
                   box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                   center_x, center_y, width, height = box.astype('int')
                   x = int(center_x - width/2)
                   y = int(center_y - height/2)
                   persons.append([x, y, int(width), int(height)])

    # Remove redundant detections using non-maximum suppression
        indices = cv2.dnn.NMSBoxes(persons, [1.0]*len(persons), confidence_threshold, nms_threshold)

    # Count number of persons
        count = len(indices)

    # Draw bounding boxes and count on frame
        for i in indices:
            if isinstance(i, list) and len(i) > 0:
               i = i[0]
            person = persons[i]
            x, y, width, height = person
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
        cv2.putText(frame, f'Person count: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

# Initialize camera
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
     cap.set(cv2.CAP_PROP_FPS, 30)

     while True:
    # Read frame from camera
        ret, frame = cap.read()

    # Process frame
        frame = process_frame(frame)                                                                                                                      

    # Display output
        cv2.imshow('Output', frame)

    # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
           break

# Release resources
     cap.release()
     cv2.destroyAllWindows()
     return render_template("campred.html",data1=camera())    

@app.route("/upload",methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No video file found"
    video=request.files['video']
    if video.filename=="":
        return "No video file selected"
    if video and allowed_file(video.filename):
        video.save('static/videos/' + video.filename)
    max_count2 = 0
    framex2 = []
    county2 = []
    max2 = []
    avg_acc2_list = []
    max_avg_acc2_list = []
    max_acc2 = 0
    max_avg_acc2 = 0
#video input
    vid = cv2.VideoCapture('static/videos/'+ video.filename)
    odapi = DetectorAPI()
    threshold = 0.7

    check, frame = vid.read()
    if check == False:
         print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
    x2 = 0
    while vid.isOpened():
    # check is True if reading was successful
        check, frame = vid.read()
        if (check == True):
            img = cv2.resize(frame, (800, 500))
            boxes, scores, classes, num = odapi.processFrame(img)
            person = 0
            acc = 0
            for i in range(len(boxes)):
            # print(boxes)
            # print(scores)
            # print(classes)
            # print(num)
            # print()
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    person += 1
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                    cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 255), 1)  # (75,0,130),
                    acc += scores[i]
                    if (scores[i] > max_acc2):
                        max_acc2 = scores[i]

            if(person>max_count2):
               max_count2=person            

      
            county2.append(person)
            x2 += 1
            framex2.append(x2)
            if (person >= 1):
                avg_acc2_list.append(acc / person)
                if ((acc / person) > max_avg_acc2):
                    max_avg_acc2 = (acc / person)
            else:
               avg_acc2_list.append(acc)

            lpc_count = person
            opc_count = max_count2
            lpc_txt = "Live Person Count: {}".format(lpc_count)
            opc_txt = "Overall Person Count:{}".format(opc_count)
            cv2.putText(img, lpc_txt, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.putText(img, opc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.imshow("Human Detection from Video", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
               break
        else:
          break

    vid.release()
    cv2.destroyAllWindows()    
    return render_template("vidprediction.html",data2=upload())     


if __name__=='__main__':
    app.run(debug=True)

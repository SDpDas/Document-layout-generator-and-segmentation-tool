import numpy as np #For all mathematical operations such as doing arrays and list
import cv2 #For reading model into python code 
import os #For modifying and checking folder properties
from pdf2image import convert_from_path #Self explanatory
import img2pdf #Self explanatory
from datetime import datetime #For doc folder timestamps

#Input File Path
pdf_file = 'C:\\Users\\SAGAR DEEP\\Desktop\\Document Layout Generation and Segmentation\\arxiv1.pdf'

#Constants for Image Size and Resolution
IMAGE_WIDTH = 2550
IMAGE_HEIGHT = 3300
IMAGE_DPI = 96

#Load YOLOv3 model
weightsPath = r'C:\\Users\\SAGAR DEEP\\Desktop\\Document Layout Generation and Segmentation\\yolo-coco\\yolov3.weights'
configPath = r'C:\\Users\\SAGAR DEEP\\Desktop\\Document Layout Generation and Segmentation\\yolo-coco\\yolov3.cfg'
labelsPath = r'C:\\Users\\SAGAR DEEP\\Desktop\\Document Layout Generation and Segmentation\\yolo-coco\\classes.names'

#Using Class Labels File
LABELS = open(labelsPath).read().strip().split("\n")

#YOLOv3 network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Object detection func.
def det_obj(image):
    (H, W) = image.shape[:2]

    #Creating output layer names
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    #Creating image blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    #Set input to network
    net.setInput(blob)

    #Forward passing through the network
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    #Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores) 
            confidence = scores[classID]

            #confidence.size checks if the confidence array is not empty
            if confidence.size > 0 and confidence > 0.3: #Adjustable as needed
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    #Applying minima suppression to weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold = 0.3, nms_threshold = 0.2)

    if len(idxs) > 0:
        for i in idxs.flatten():

            if i >= len(boxes) or i >= len(confidences) or i >= len(classIDs):
                print(f"Index {i} out of range. Boxes: {len(boxes)}, Confidences: {len(confidences)}, ClassIDs: {len(classIDs)}")
                continue

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = (255, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

#Resizes each image according to GUI window size
def resize_image(image, max_width, max_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_aspect_ratio = max_width / max_height

    if aspect_ratio > target_aspect_ratio:
        new_w = max_width 
        new_h = int(new_w / aspect_ratio)

    else:
        new_h = max_height
        new_w = int(new_h * aspect_ratio)

    resize_image = cv2.resize(image, (new_w, new_h))
    return resize_image

#Creates new document folder for every new document
def create_document_folder():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    doc_folder = os.path.join('Processed Files', f'document_{timestamp}')
    os.makedirs(doc_folder, exist_ok=True)
    return doc_folder

#Converting PDF to images and call object detection 
def pdf_to_image_and_detect(pdf_file):

    doc_folder = create_document_folder() 

    input_folder = os.path.join(doc_folder, 'input_img')
    output_folder = os.path.join(doc_folder, 'output_img')
    output_pdf_folder = os.path.join(doc_folder, 'output_pdf')

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_pdf_folder, exist_ok=True)

    images = convert_from_path(pdf_file, dpi=IMAGE_DPI)

    pdf_pages = []

    for i, image in enumerate(images):
        #Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        #Save input image to Input folder
        input_path = os.path.join(input_folder, f'page_{i+1}.jpg')
        cv2.imwrite(input_path, image)

        #Adjust the size of image if needed
        max_width = 1200
        max_height = 800
        image = resize_image(image, max_width, max_height)

        #Perform object detection
        output_image = det_obj(image)

        #Save annotated image to output folder
        output_path = os.path.join(output_folder, f'page_{i+1}.jpg')
        cv2.imwrite(output_path, image)

        #Append output images to pdf_pages for PDF conversion
        pdf_pages.append(output_path)

        #Showing image results one by one
        cv2.imshow(f'Page {i+1}', output_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    #Converts all attached images to single PDF
    output_pdf_path = os.path.join(output_pdf_folder, 'Output.pdf')
    with open(output_pdf_path, 'wb') as f:
            f.write(img2pdf.convert(pdf_pages))
            
    print(f"PDF generated: {output_pdf_path}")

#Passes input PDF to object detection and processing function
pdf_to_image_and_detect(pdf_file)



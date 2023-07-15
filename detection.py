from imageai.Detection import ObjectDetection  # this is 
import os

execution_path = os.getcwd() # the os getting the current working directory
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,"retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path , "AhmadHammad.jpeg"),
    output_image_path=os.path.join(execution_path , "imagenew.jpeg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"] )
    print("--------------------------------")
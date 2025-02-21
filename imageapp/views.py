from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image
import cv2  
import numpy as np
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
def get_label(text):
    if text <= 9:
        return int(text)
    else:
        match text :
            case 10: return'b'
            case 11: return'c'
            case 12: return'd'
            case 13: return'g'
            case 14: return'h'
            case 15: return'j'
            case 16: return'l'
            case 17: return'm'
            case 18: return'n'
            case 19: return's'
            case 20: return't'
            case 21: return'v'
            case 22: return'y'
    return 'error' 

model = YOLO(BASE_DIR/'text_detection'/'best.pt')
model_plate=YOLO(BASE_DIR/'plate_detection'/'best.pt')

def process_image(image_path):
    show = cv2.imread(image_path)
    results = model_plate.predict(show)
    boxes = results[0].boxes.xyxy.cpu().tolist() 
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  
        cropped_object = show[y1:y2, x1:x2]  
        cropped_object = cv2.resize(cropped_object, (450, 200))
    final = process_text(cropped_object)
    original_height, original_width = show.shape[:2]
    final_resized = cv2.resize(final, (original_width, 200))
    if len(show.shape) == 2:  
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
    if len(final_resized.shape) == 2:
        final_resized = cv2.cvtColor(final_resized, cv2.COLOR_GRAY2BGR)
    combined_height = original_height + final_resized.shape[0]
    combined_image = np.zeros((combined_height, original_width, 3), dtype=np.uint8)
    combined_image[0:original_height, :] = show
    combined_image[original_height:, :] = final_resized
    
    return combined_image
    
  
   
     
def process_text(image):
    results = model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    prec = results[0].boxes.conf.cpu().tolist()
    labels = results[0].boxes.cls.cpu().tolist()
    final_plate = image.copy()
    for idx, (box, confidence, label) in enumerate(zip(boxes, prec, labels)):
        if confidence > 0.4:  
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(final_plate, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(final_plate, f'{get_label(label)}', (int(x1), int(y1)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return final_plate
    
def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            processed_image = process_image(image_instance.image.path)
            cv2.imwrite(image_instance.image.path, processed_image)
            image_instance.save()
            return redirect('home')
    else:
        form = ImageUploadForm()
    images = Image.objects.all()
    return render(request, 'imageapp/home.html', {'form': form, 'images': images})

def delete_image(request, image_id):
    image = Image.objects.get(id=image_id)
    image.delete()
    return redirect('home')
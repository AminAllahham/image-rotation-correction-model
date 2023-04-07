import os
import random
import uuid
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image



def rotate_image(image, angle): 
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result

def pdf_to_rotated_images(pdf_path, pdf_name, rotated_images_folder):
    images = convert_from_path(pdf_path + '/' + pdf_name)
    os.makedirs(rotated_images_folder, exist_ok=True)
    results = []
    print('Total images: ', len(images))

    
    for i, image in enumerate(images):
        print('Processing image: ', i)
        imageFileName = pdf_name + '-' + str(i) + '-' + str(uuid.uuid4().hex)[:8] + '.jpg'

        if np.mean(image) == 255:
            print('Skipping blank image: ', i)
            continue

         
        for  copyIdx in range(0, 30):
             index = copyIdx + 1
             idxToUniform = index < 16 and -index or index - 15
             nextIndex =  idxToUniform + 1

             degree = copyIdx  == 0 and 0 or random.uniform(nextIndex, idxToUniform)
             print('Processing copy image: ', copyIdx, ' with degree: ', degree)


             imageFileName = pdf_name + '-' + str(i) + '-' + str(uuid.uuid4().hex)[:8]  + '(' + str(degree) + ')' + '.jpg'
    
             result = rotate_image(np.array(image), degree)
             resultAsImage = Image.fromarray(result)
             resultImagePath = os.path.join(rotated_images_folder, imageFileName)       

             row = {
                'rotated_path': resultImagePath,
                'degree': degree
             }
             results.append(row)

             resultAsImage.save(resultImagePath, 'JPEG')
             
    return results
        
    


if not os.path.exists('training-data'):
      os.makedirs('training-data')


def generate():
     for pathTo in ["training-pdfs"]:
      resultsList = []   
      data_folder_name = pathTo.split('-')[0] + '-data'

      rotated_images_folder  = data_folder_name + '/' + 'rotated-images'

      for pdf_name in os.listdir(pathTo):
          print('Processing pdf: ', pdf_name)
          # if not pdf name ends with pdf, skip
          if not pdf_name.endswith('.pdf'):
                continue

          values = pdf_to_rotated_images(pathTo, pdf_name, rotated_images_folder)
          resultsList.extend(values)
          

      print('Total images: ', len(resultsList))
      return resultsList


allData = generate()

allDataScuffled = random.sample(allData, len(allData))

trainingData = allDataScuffled[:int(len(allDataScuffled) * 0.4)]
validationData = allDataScuffled[int(len(allDataScuffled) * 0.4):int(len(allDataScuffled) * 0.8)]
testingData = allDataScuffled[int(len(allDataScuffled) * 0.8):]


trainingDataDf = pd.DataFrame(trainingData)
trainingDataDf.to_csv('training-data/training-data.csv', index=False)

validationDataDf = pd.DataFrame(validationData)

validationDataDf.to_csv('training-data/validation-data.csv', index=False)

testingDataDf = pd.DataFrame(testingData)

testingDataDf.to_csv('training-data/testing-data.csv', index=False)



print('Total training images: ', len(trainingData))
print('Total validation images: ', len(validationData))
print('Total testing images: ', len(testingData))





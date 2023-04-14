import os
import random
import uuid
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path



def rotate_image(image, angle): 
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result

    
def load_pdf(pdf_path):
    print('Loading pdf: ', pdf_path)
    images = convert_from_path(pdf_path)
    return images


def create_data_folder(images, folder_name , num_copies=30):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    if not os.path.exists(os.path.join(folder_name, 'rotated-images')):
        os.mkdir(os.path.join(folder_name, 'rotated-images'))
        
    csv_rows = []
    for image in images:
        for  copyIdx in range(0, num_copies):
             index = copyIdx + 1
             idxToUniform = index < 16 and -index or index - 15
             nextIndex =  idxToUniform + 1

             degree = copyIdx  == 0 and 0 or random.uniform(nextIndex, idxToUniform)
            
             rotated_image = rotate_image(np.array(image), degree)

             image_name = str(uuid.uuid4()) + '.jpg' 
             image_path = os.path.join(folder_name + '/rotated-images', image_name)
             cv2.imwrite(image_path, rotated_image)
             csv_rows.append([image_path, degree])

        df = pd.DataFrame(csv_rows, columns=['image-path', 'degree'])
        df.to_csv(os.path.join(folder_name, 'dataset.csv'), index=False)


if __name__ == "__main__":
    pdfs = [f for f in os.listdir('training-pdfs') if f.endswith('.pdf')]
    all_images = []
    for pdf in pdfs:
        images = load_pdf(os.path.join('training-pdfs', pdf))
    
        for image in images:
            all_images.append(image)


    all_images = random.sample(all_images, len(all_images))

    print('total images before copies: ', len(all_images))

    # split all images into 40% train, 20% test, 40% validation
    train_images = all_images[:int(len(all_images) * 0.4)]
    test_images = all_images[int(len(all_images) * 0.4):int(len(all_images) * 0.6)]
    validation_images = all_images[int(len(all_images) * 0.6):]

    create_data_folder(train_images, 'training-data')

    create_data_folder(test_images, 'testing-data')

    create_data_folder(validation_images, 'validation-data')

        
    

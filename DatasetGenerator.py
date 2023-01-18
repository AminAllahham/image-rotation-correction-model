import csv
import os
import random
import uuid

import cv2
import numpy as np


class RadomImagesRotationDatasetGenerator:
    def __init__(self, original_images_path, rotated_images_path, rotation_degree ,csv_file_path):
        self.original_images_path = original_images_path
        self.rotated_images_path = rotated_images_path
        self.csv_file_path = csv_file_path
        self.rotation_degree = rotation_degree

    def create_folder(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

    def read_folder_images(self, folderPath):
        images = []
        for filename in os.listdir(folderPath):
            images.append(filename)
        return images

    def rotate_image(self, image, angle):

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result

    def create_copies_of_image_with_random_rotation(self, imagePath, copiesFolderName):

        result = []
        img = cv2.imread(imagePath)
        self.create_folder(copiesFolderName)

        for i in range(0, 20):
            # rotate the img random degree from -20 to 20
            degree = random.uniform(self.rotation_degree[0], self.rotation_degree[1])
            print(degree)
            rotated_img = self.rotate_image(img, degree)
            random_id = uuid.uuid1()

            copyImagePath = f'{copiesFolderName}/'+random_id.hex+'.jpg'
            cv2.imwrite(copyImagePath, rotated_img)
            result.append([imagePath, copyImagePath, degree])

        return result

    def read_folder_images_and_create_random_rotation_instance(self, form_path, to_path):

        images_from_folder = self.read_folder_images(form_path);
        images_paths = []

        for image in images_from_folder:
            image_path = f'original_images/{image}'
            values = self.create_copies_of_image_with_random_rotation(image_path, to_path)

            images_paths.append(values)

        return  images_paths

    def create_csv_file(self, rows):
        csv_rows = []
        for imageCopies in rows:
            for image in imageCopies:
                csv_rows.append(image)
        
        

        with open(self.csv_file_path, 'w') as csvFile:
            fieldnames = ['original_image_path', 'changed_rotation_image_path', 'rotation_degree']
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()

            for row in csv_rows:
                writer.writerow({
                    'original_image_path': row[0],
                    'changed_rotation_image_path': row[1],
                    'rotation_degree': row[2]
                })

            print('Created a csv file with' , len(csv_rows), 'rows')




# ## run

# original_images_path = 'original_images'
# rotated_images_path = 'rotated_images'
# csv_file_path = 'images-dataset.csv'


# Generator = RadomImagesRotationDatasetGenerator(original_images_path, rotated_images_path, [-20, 20], csv_file_path)

# images = Generator.read_folder_images_and_create_random_rotation_instance(original_images_path, rotated_images_path)

# Generator.create_csv_file(images)



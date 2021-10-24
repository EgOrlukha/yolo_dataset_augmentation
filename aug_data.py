from augmentation_instruments import pic_augmentation as aug
from augmentation_instruments import showImage, saveImage
from augmentation_instruments import yoloRotatebbox

import glob
import os
import random
import shutil
from functools import partial
from sys import argv
from tqdm import tqdm

def get_img_data(img_path):
    file_root, file_name = os.path.split(img_path)
    img_name, img_ext = os.path.splitext(file_name)
    return file_root, img_name, img_ext

def isPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def apply_augmentation(img_path, aug_type):
    """
    Применить аугментацию к изображению, если тип аугментации доступен
    и если есть файл *.txt с разметкой

    @img_path: относительный путь к изображению
    @aug_type: тип аугментации из числа "deblur", "gaus_blur", "sp_noise", "brightness", "contrast"
    """
    image = aug(img_path)

    img_root, image_name, img_ext = get_img_data(img_path)

    brightness_level = random.choice([-150, -75, 75, 150])
    contrast_level = random.choice([-64, 64])
    mask_size_level = random.choice([7,9,11,13])
    sp_noise_level = random.uniform(0.04,0.13)

    angles = [30,60,90,120,150,180,210,240,270,300,330]

    augmentation = {"deblur":partial(image.deblurring), 
                    "gaus_blur":partial(image.gaussian_blurring, mask_size=mask_size_level), 
                    "sp_noise":partial(image.sp_noise, sp_noise_level), 
                    "brightness":partial(image.apply_brightness_contrast, brightness=brightness_level), 
                    "contrast":partial(image.apply_brightness_contrast, contrast=contrast_level)}     
    
    augmented_image = augmentation[aug_type]()

    isPath(os.path.join(img_root,aug_type))
    # showImage(augmented_image)

    saveImage(augmented_image,
                os.path.join(img_root,aug_type),
                image_name,
                suffix="_"+aug_type)

    # пересохранить файл разметки с новым именем в папку к изображению
    shutil.copy(os.path.join(img_root,image_name)+".txt",
                os.path.join(img_root,aug_type,image_name)+"_"+aug_type+".txt")

    # поворот аугментированного изображения на все углы
    for angle in angles:
        im = yoloRotatebbox(os.path.join(img_root,aug_type,image_name)+"_"+aug_type, img_ext, angle)

        rotated_bbox = im.rotateYolobbox()
        rotated_image = im.rotate_image()

        saveImage(rotated_image,
            os.path.join(img_root,aug_type),
            image_name,
            suffix="_"+aug_type+"_"+"rot"+str(angle))

        bbox_file_name = os.path.join(img_root,aug_type,image_name)+"_"+aug_type+"_" + "rot" + str(angle) + '.txt'
        if os.path.exists(bbox_file_name):
                os.remove(bbox_file_name)
        # to write the new rotated bboxes to file
        for i in rotated_bbox:
            with open(bbox_file_name, 'a') as fout:
                fout.writelines(
                    ' '.join(map(str, im.cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')

if __name__ == "__main__":
    # открываем папку с изображениями
    _, path = argv
    # перечисляем изображения в папке
    all_files = glob.glob(os.path.join(path,"*.jpg"))
    # создаём список возможных аугментаций
    augmentation_types = {0:"deblur", 1:"gaus_blur", 2:"sp_noise", 3:"brightness", 4:"contrast"}

    # для каждого изображения в папке применим аугментацию
    print("Augmentation is in progress...")
    for img in tqdm(all_files):
        # проверка на наличие разметки в данных
        img_root, image_name, _ = get_img_data(img)
        if os.path.exists(os.path.join(img_root,image_name)+".txt"): 
            # аугментация всех видов из списка
            for aug_type in augmentation_types.values():
                apply_augmentation(img, aug_type)
        else:
            print("\nNo labels for the picture '"+image_name+".jpg'")

    # # повороты для каждого дополненного изображения
    # print("Rotating all augmented images...")
    
    # augmented_files = glob.glob(os.path.join(path,"*.jpg"))
    
    # for img in tqdm(augmented_files):
    #     # проверка на наличие разметки в данных
    #     img_root, image_name, img_ext = get_img_data(img)
    #     angles = [30,60,90,120,150,180,210,240,270,300,330]
        
    #     if os.path.exists(os.path.join(img_root,image_name)+".txt"): 
    #         # повороты на все углы из списка
    #         for angle in angles:
    #             im = yoloRotatebbox(os.path.join(img_root,image_name), img_ext, angle)

    #             rotated_bbox = im.rotateYolobbox()
    #             rotated_image = im.rotate_image()

    #             saveImage(rotated_image,
    #                 img_root,
    #                 image_name,
    #                 suffix="_"+"rot"+str(angle))

    #             bbox_file_name = os.path.join(img_root,image_name)+"_" + "rot" + str(angle) + '.txt'
    #             if os.path.exists(bbox_file_name):
    #                 os.remove(bbox_file_name)
    #             # to write the new rotated bboxes to file
    #             for i in rotated_bbox:
    #                 with open(bbox_file_name, 'a') as fout:
    #                     fout.writelines(
    #                         ' '.join(map(str, im.cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')

    #     else:
    #         print("\nNo labels for the picture '"+image_name+".jpg'")
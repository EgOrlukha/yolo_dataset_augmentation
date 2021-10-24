#!/usr/bin/python

import cv2
import numpy as np
import random
import os


def showImage(image, image_name="Image"):
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveImage(image, path, name, suffix=""):
    """
    Сохранить изображение в *.jpg формате
    """
    try:
        isWritten = cv2.imwrite(os.path.join(path,name)+suffix+".jpg", image)
    except Exception as error:
        print('Can not save image: ' + repr(error))


class pic_augmentation(object):
    """
    Инструменты для аугментации изображений:
    резкость
    размытие (среднее, гауссовое, медианное)
    шум "соль-перец"
    яркость
    контрастность
    
    @image_path: путь доступа к изображению, с которым хотим работать
    """
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def deblurring(self):
        """
        Резкость
        https://medium.com/analytics-vidhya/data-augmentation-techniques-using-opencv-657bcb9cc30b
        """
        kernel = np.array([[-1, -1, -1], [-1, 8.7, -1], [-1, -1, -1]])
        image = cv2.filter2D(self.image, -1, kernel)
        return image

    def averaging_blurring(self, mask_size=3):
        """
        Размытие по средней
        https://habr.com/ru/post/539228/

        @mask_size: размерность стороны квадратного фильтра
        """
        return cv2.blur(self.image, (mask_size, mask_size))

    def gaussian_blurring(self, mask_size=3):
        """
        Размытие по Гауссу
        https://habr.com/ru/post/539228/

        @mask_size: размерность стороны квадратного фильтра
        """
        return cv2.GaussianBlur(self.image, (mask_size, mask_size), 0)

    def median_blurring(self, mask_size=3):
        """
        Размытие по медиане
        https://habr.com/ru/post/539228/

        @mask_size: размерность стороны квадратного фильтра
        """
        return cv2.medianBlur(self.image, mask_size)

    def sp_noise(self, prob):
        """
        Шум "соль-перец"
        https://dev-gang.ru/article/vvedenie-v-obrabotku-izobrazhenii-v-python-s-opencv-bpvt25yc6e/

        @prob: Probability of the noise
        """
        output = np.zeros(self.image.shape, np.uint8)
        thres = 1 - prob
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        return output

    def apply_brightness_contrast(self, brightness=0, contrast=0):
        """
        Контрастность и яркость в одном флаконе
        https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv

        @brightness: яркость изображения. Рабочий диапазон [-127;127]
        @contrast: контрастность изображения. Рабочий диапазон [-64;64]
        """
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(self.image, alpha_b, self.image, 0, gamma_b)
        else:
            buf = self.image.copy()

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
              
class yoloRotatebbox(object):
    def __init__(self, filename, image_ext, angle):
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def yoloFormattocv(self, x1, y1, x2, y2, H, W):
        """
        Convert from Yolo_mark to opencv format
        """
        bbox_width = x2 * W
        bbox_height = y2 * H
        center_x = x1 * W
        center_y = y1 * H

        voc = []

        voc.append(center_x - (bbox_width / 2))
        voc.append(center_y - (bbox_height / 2))
        voc.append(center_x + (bbox_width / 2))
        voc.append(center_y + (bbox_height / 2))

        return [int(v) for v in voc]

    def cvFormattoYolo(self, corner, H, W):
        """
        Convert from opencv format to yolo format
        H,W is the image height and width
        """
        bbox_W = corner[3] - corner[1]
        bbox_H = corner[4] - corner[2]

        center_bbox_x = (corner[1] + corner[3]) / 2
        center_bbox_y = (corner[2] + corner[4]) / 2

        return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,
                                                                                                                6)
    
    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]

        f = open(self.filename + '.txt', 'r')

        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = self.yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                            float(bbox[3]), float(bbox[4]), H, W)

                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                        lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        
        return rotated_mat



if __name__ == "__main__":
    image = pic_augmentation("pics/000002.jpg")
    deblured_img = image.deblurring()
    gaussian_blur_img = image.gaussian_blurring(mask_size=7)
    pepper_noised_img = image.sp_noise(0.01)
    showImage(pepper_noised_img, image_name="Image")
    saveImage(pepper_noised_img, "pics", "RandomName")
    bright_contr_img = image.apply_brightness_contrast(brightness=0, contrast=0)
    showImage(bright_contr_img, image_name="Image")
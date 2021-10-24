# YOLO Dataset Augmentation lib
## Инструкция по использованию этой библиотеки

Запуск всех файлов осуществлять из консоли.

### **GoogleCrawl_to_Dataset.py** 
Это скрипт для скачивания фотографий из Google для дополнения датасета.

* Формат вызова: **python GoogleCrawl_to_Dataset.py [PATH/TO/SAVE/IMAGES] [keyword] [number of images]**

### **augmentation_instruments.py** 
Cодержит класс с инструментами аугментации данных.

### **aug_data.py** 
Это скрипт для аугментации фотографий из указанной папки.

Будут дополнены только фото с разметкой labelimg. 

Параметры для аугментации: резкость, размытие по Гауссу, яркость, контрастность, шум "соль-перец".

Каждое дополненное изображение будет также повёрнуто на 30, 60, ..., 330 градусов с сохранением разметки.

Формат вызова: **python aug_data.py [PATH/TO/IMAGES]**
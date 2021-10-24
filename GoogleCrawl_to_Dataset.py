"""
Этот файл позволит скачать фотографии из Google по запросу.
Так мы можем дополнять датасеты, если будет слишком мало фотографий.
"""

from icrawler.builtin import GoogleImageCrawler
from sys import argv

# #абсолютный путь к папке, куда будем загружать
# dir_pics = 'C:/Users/Пользователь/Desktop/TEST/pics'
# #сколько фотографий надо загрузить
# num_pics = 10



if __name__ == "__main__":
    _, path2save, zapros, num_pics = argv
    google_crawler = GoogleImageCrawler(storage={'root_dir': path2save})
    google_crawler.crawl(keyword=zapros, max_num=int(num_pics))
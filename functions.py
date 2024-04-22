# Библиотека компьютерного зрения
import cv2
# Импортируем библиотеку для работы с операционной системой
import os
# Импортируем библиотеку для работы с датой
from datetime import date
from datetime import datetime
# Импортируем библиотеку для работы с массивами
import numpy as np
# Импортируем библиотеку для работы с алгоритмом k-ближайших соседей
from sklearn.neighbors import KNeighborsClassifier
# Импортируем библиотеку для работы с таблицами
import pandas as pd
# Импортируем библиотеку для сохранения и загрузки модели
import joblib
# Импортируем библиотеку для работы с CSV-файлами
import csv


datetoday = date.today().strftime("%d-%B-%Y")

# Создаем объект для обнаружения лиц на изображении
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Создаем объект для работы с видеопотоком с камеры
cap = cv2.VideoCapture(0)


# Создаем директории и csv-файл
def mkdirs():
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    if not os.path.isdir('static'):
        os.makedirs('static')
    if not os.path.isdir('static/faces'):
        os.makedirs('static/faces')
    if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["LastName", "Name", "Patronymic", "id", "Time"])

# Число зарегестрированных пользователей
def totalreg():
    return len(os.listdir('static/faces'))


# Достаем координаты лица из изображения
def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 7)
        return face_points
    
    return []

# Распознаем лицо
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# Обучение модели на лицах из каталога static/faces
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Извлекаем данные о сегодняшней посещаемости
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    lastnames = df['LastName']
    names = df['Name']
    patronymics = df['Patronymic']
    id = df['id']
    times = df['Time']
    l = len(df)
    return lastnames, names, patronymics, id, times, l


# Добавляем запись о посещении человека в csv-файл
def add_attendance(name):
    userlastname = name.split('_')[0]
    username = name.split('_')[1]
    userpatronymic = name.split('_')[2]
    userid = name.split('_')[3]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['id']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{userlastname}, {username}, {userpatronymic}, {userid}, {current_time}')

## ID
def which_id():
    userlist = os.listdir('static/faces')
    id = 0
    if len(userlist):
        for user in userlist:
            if int(user.split('_')[3]) >= id:
                id = int(user.split('_')[3]) + 1
    return id
    
# Библиотека компьютерного зрения
import cv2
# Импортируем библиотеку для работы с операционной системой
import os
# Импортируем библиотеку для создания веб-приложения
from flask import Flask, request, render_template
# Импортируем функции из файла functions.py
from functions import *

app = Flask(__name__)

# Главная страница
@app.route('/')
def home():
    lastnames, names, patronymics, id, times, l = extract_attendance()    
    return render_template('home.html',
                            lastnames=lastnames, 
                            names=names,
                            patronymics = patronymics,
                            id=id, times=times, l=l, totalreg=totalreg(), datetoday=datetoday) 


# Функция запускается, при нажатии на кнопку "Отметиться"
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',
                                totalreg=totalreg(), 
                                datetoday=datetoday, 
                                mess='Добавьте пользователя.') 

    while True:
        frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    lastnames, names, patronymics, id, times, l = extract_attendance()    
    return render_template('home.html',
                            lastnames = lastnames, names=names,
                            patronymics = patronymics, id=id, 
                            times=times, l=l, totalreg=totalreg(), datetoday=datetoday) 


# Функция запускается при добавлении нового пользователя
@app.route('/add',methods=['GET','POST'])
def add():
    newuserlastname = request.form['newuserlastname']
    newusername = request.form['newusername']
    newuserpatronymic = request.form['newuserpatronymic']
    newuserid = which_id()
    userimagefolder = 'static/faces/' + newuserlastname + '_' +  newusername + '_' + newuserpatronymic + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y : y + h, x : x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    lastnames, names, patronymics, id, times, l = extract_attendance()    
    return render_template('home.html', lastnames=lastnames, names=names, patronymics = patronymics,
                           id=id, times=times, l=l, totalreg=totalreg(), datetoday=datetoday) 

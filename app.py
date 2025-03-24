import decimal
import os
import datetime
import cv2
from flask import Flask, request, jsonify, render_template

from PIL import Image, ExifTags

from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename
import torch
import cv2
import sqlite3
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

#Flask 앱 초기화
app = Flask(__name__)

#.env 로드
load_dotenv()

#Roboflow Inference API 설정
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= os.environ.get('ROBOFLOW2_API_KEY')
)

#이미지 저장 폴더 생성(폴더가 없을 경우)
IMAGE_FOLDER = 'location'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

#SQLite 데이터베이스 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'image_data.db')

#데이터베이스 초기화 및 테이블 생성 함수
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                datetime TEXT,
                latitude REAL,
                longitude REAL,
                label TEXT,
                confidence REAL
            )
        """)
        conn.commit()
        print("Database initialized and table created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()  # 롤백을 통해 데이터베이스 상태를 이전 상태로 복구
    finally:
        if conn:
            conn.close()
              
#Flask 요청 전 데이터베이스 초기화
@app.before_request
def setup():
    init_db()

#이미지 데이터베이스에 저장하는 함수
def save_image_data(filename, datetime, latitude, longitude, label, confidence):
    try:
        conn = sqlite3.connect('image_data.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO images (filename, datetime, latitude, longitude, label, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, datetime, latitude, longitude, label, confidence))
        conn.commit()
    except Exception as e:
        print("An error occurred:", e)
    finally:
        conn.close()

#홈페이지 렌더
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    temp_path = os.path.join('test', 'kick.jpg')
    print(temp_path)
    image = Image.open(temp_path)
    # exif_data = image._getexif()
    img = cv2.imread(temp_path)
    h, w = img.shape[:2]
    scale = 300 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    result_kickboard = CLIENT.infer(img, model_id="kickboard-22-jt3v1/1")
    print('result: ', result_kickboard)

    result_person = CLIENT.infer(img, model_id="person-469rx-3u095/1")
    print('result: ', result_person)

    result_helmet = CLIENT.infer(img, model_id="helmet-nw6lg-i02zn/1")
    print('result: ', result_helmet)

    img = object_detection(result_kickboard['predictions'], img)
    img = object_detection(result_person['predictions'], img)
    img = object_detection(result_helmet['predictions'], img)

    save_path = os.path.join('test2', 'testimg.jpg')
    cv2.imwrite(save_path, img)
    print(1)
    return render_template('index.html')

#yolo 객체 탐지
@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    #temp에 탐지할 이미지 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    file.save(temp_path)

    image = Image.open(temp_path)
    
    #temp에 넣은 이미지의 EXIF 데이터에서 위도 경도 추출
    lat, lon = get_image_location(image)
    
    #temp에 넣은 이미지 전처리
    img = cv2.imread(temp_path)
    h, w = img.shape[:2]
    scale = 300 / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # if not lat or not lon:
    #     os.remove(temp_path)
    #     return jsonify({'error': 'No location data available, cannot upload the photo.'}), 400

    result_kickboard = CLIENT.infer(img, model_id="kickboard-22-jt3v1/1")
    print('result: ', result_kickboard)


    #os.remove(temp_path)

    helmet_status = None
    traffic_violation_detection = '위반사항 없음'

    #헬멧 착용여부 판단
    if any(item['confidence'] > 0.8 for item in result_kickboard['predictions']) :
        img = object_detection(result_kickboard['predictions'], img)

        result_person = CLIENT.infer(img, model_id="person-469rx-3u095/1")
        print('result: ', result_person)


        if any(item['confidence'] > 0.8 for item in result_person['predictions']) :
            img = object_detection(result_person['predictions'], img)

            result_helmet = CLIENT.infer(img, model_id="helmet-nw6lg-i02zn/1")
            print('result: ', result_helmet)

            if any(item['confidence'] > 0.8 for item in result_helmet['predictions'])  :
                helmet_status = '착용'
                img = object_detection(result_helmet['predictions'], img)
            else:
                helmet_status = '미착용'
        else:
            traffic_violation_detection = '사람 감지 실패'
            return jsonify({'위반 감지': traffic_violation_detection})
    else:
        traffic_violation_detection = '킥보드 감지 실패'
        return jsonify({'위반 감지': traffic_violation_detection})
    
    
    #헬멧 미착용 시 static에 사진데이터 저장 / 함수화 예정
    if helmet_status == '미착용':
        '''save_filename = f"{filename[:-4]}_lat{lat}_lon{lon}_time{timestamp}.jpg"
        save_path = os.path.join(IMAGE_FOLDER, save_filename)'''
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        save_image_data(filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lat, lon, "label", "0.1")
        save_path = os.path.join('static/img/noHelmet', filename)
        cv2.imwrite(save_path, img)
        return jsonify({'helmet_status': helmet_status})
    elif helmet_status == '착용':
        '''save_filename = f"{filename[:-4]}_lat{lat}_lon{lon}_time{timestamp}.jpg"
        save_path = os.path.join(IMAGE_FOLDER, save_filename)'''
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        save_image_data(filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lat, lon, "label", "0.1")
        save_path = os.path.join('static/img/Helmet', filename)
        cv2.imwrite(save_path, img)
        return jsonify({'helmet_status': helmet_status})
    else :
        '''save_filename = f"{filename[:-4]}_lat{lat}_lon{lon}_time{timestamp}.jpg"
        save_path = os.path.join(IMAGE_FOLDER, save_filename)'''
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        save_image_data(filename, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), lat, lon, "label", "0.1")
        save_path = os.path.join('static/img/notHelmet', filename)
        cv2.imwrite(save_path, img)
        return jsonify({'helmet_status': helmet_status})

#이미지 EXIF 데이터에서 위도 경도 가져오는 함수
def get_image_location(image):
    """Extract GPS coordinates from an image's EXIF data."""
    exif_data = image._getexif()
    if not exif_data:
        return None, None

    gps_info = exif_data.get(34853)
    if not gps_info:
        return None, None

    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    lat_ref = gps_info.get(1)
    lon_ref = gps_info.get(3)
    lat = gps_info.get(2)
    lon = gps_info.get(4)
    if lat and lon:
        lat = convert_to_degrees(lat)
        lon = convert_to_degrees(lon)
        if lat_ref != 'N':
            lat = -lat
        if lon_ref != 'E':
            lon = -lon
        return lat, lon
    return None, None

def object_detection(predictions, img):
    for prediction in predictions:
        centerx = int(prediction['x'])
        centery = int(prediction['y'])
        symmetric = int(prediction['width'])/2
        horizontal = int(prediction['height'])/2
        
        x1 = int(centerx - symmetric)
        y1 = int(centery - horizontal)
        x2 =  int(centerx + symmetric)
        y2 =  int(centery + horizontal)
        
        label = prediction['class']
        #label = str('Helmet')
        conf = prediction['confidence']
        #conf = 0.84
        text = str(label) + ' ' + str(conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, text, (x1+5, y1+20 ), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 1)
    return img


#객체 분할 후 탐지된 객체만 남기고 삭제 / 현재 사용 X
def draw_segmented_objects(image, contours, label_cnt_idx, bubbles_count):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image



if __name__ == '__main__':
    print("Initializing database...")
    init_db()  # Ensure database is initialized before starting the app
    print("Starting Flask app...")
    app.run(debug=True)
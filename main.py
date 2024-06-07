from flask import Flask, request, render_template, redirect, url_for
import paddlehub as hub
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 创建上传文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 加载 PaddleHub OCR 模型
ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # 检查文件类型
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            if img is not None:
                results = ocr.recognize_text(
                    images=[img],
                    use_gpu=False,
                    output_dir='ocr_result',
                    visualization=True,
                    box_thresh=0.5,
                    text_thresh=0.5
                )
                texts = [info['text'] for result in results for info in result['data']]
                return render_template('result.html', texts=texts)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

#!/usr/bin/env python3

import base64
from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image
from sklearn import datasets, svm
import struct


digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data, digits.target) # 1. train
app = Flask(__name__, static_folder='')


@app.route('/')
def root():
    return app.send_static_file('ocr-draw.html')


@app.route('/ocr-char', methods=['POST'])
def content():
    img_base64 = request.values.get('img')
    img = to_float_img(img_base64.partition(',')[2])
    return '%s' % clf.predict([img])[0] # 2. predict


def to_float_img(img_base64):
    # use the PNG alpha mask for RGB color; extract R and use as 0-16 grayscale
    data = base64.b64decode(img_base64)
    bmp = BytesIO()
    png = Image.open(BytesIO(data))
    white_bkg = Image.new('RGB', png.size, (255,255,255))
    white_bkg.paste(png, mask=png.split()[3])
    white_bkg.convert('RGB').save(bmp, 'BMP')
    out = bmp.getvalue()
    w = struct.unpack("<L", out[18:22])[0]
    h = struct.unpack("<L", out[22:26])[0]
    return [(255-b)*16/255 for i in range(h-1,-1,-1) for b in out[54+i*w*3:54+(i+1)*w*3:3]]


if __name__ == '__main__':
    import time, threading, webbrowser
    def delayed_browse():
        time.sleep(1)
        webbrowser.open_new_tab('http://localhost:9001/')
    threading.Thread(target=delayed_browse).start()
    app.run(host='0.0.0.0', port=9001, threaded=True)

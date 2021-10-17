import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import scipy
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import euclidean,cosine
import matplotlib.image as mpimg
import joblib
star = cv.xfeatures2d.SURF_create()
sift = sift = cv.xfeatures2d.SIFT_create()
directory='./mydataset'
img_dirs=['stop_sign','sunflower','airplanes','brain','hawksbill','watch']

o = open('images.pkl', 'rb')
final_images = pickle.load(o)
cen = open('clustered_2000.pkl', 'rb')
centers = pickle.load(cen)
topic_dist = joblib.load('topic_dist.jl')
lda = joblib.load('lda_model.jl')

app = Flask(__name__,static_url_path = "/mydataset", static_folder = "mydataset")
UPLOAD_FOLDER = './mydataset/test'
# IMAGE_FOLDER = '../random'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/uploader',methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      f.save(path)
      # npimg = np.fromstring(f, np.uint8)
      # query_img = cv.imdecode(npimg, cv.IMREAD_UNCHANGED)
      query_img = cv.imread(path)


## put this before the for i,_sorted loop
      if (type(query_img)!=type(None)):
          query_img_kps=star.detect(query_img)
          query_kp, query_des = sift.compute(query_img,query_img_kps)



          FLANN_INDEX_KDTREE = 0
          index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
          search_params = dict(checks=50)

          flann = cv.FlannBasedMatcher(index_params,search_params)
          query_arr=[]
          matches = flann.knnMatch(np.asarray(query_des,np.float32),np.asarray(centers,np.float32),k=1)
          query_arr.append([matches[j][0].trainIdx for j in range(len(matches))])

          dictionary_size=500
          q={}
          query_hist=[0]*dictionary_size
          for j in range(len(query_arr[0])):
              if (query_arr[0][j] in q):
                  q[query_arr[0][j]]+=1
              else:
                  q[query_arr[0][j]]=1
          for k in q:
             query_hist[k]=q[k]

          query_hist=np.array(query_hist).reshape(1,-1)

          query_topic = lda.transform(query_hist)

          euc=[]
          for i in range(len(topic_dist)):
              euc.append([final_images[i],cosine(query_topic[0],topic_dist[i])])

          img_sorted = sorted(euc, key=lambda x: x[1])

          files = []
          count = 0
          files.append('test/'+filename)

          for i,_ in img_sorted:
            if count>=10:
                break
            l = i.split('/')
            files.append(l[2]+'/'+l[3])
            count+=1
          return render_template('upload.html',files = files)
      else:
          return 'Upload a valid image!'
if __name__ == '__main__':
   app.run(debug = True)

#api_sklearn.py
import hug
#その他必要なライブラリをインポートする
import keras
import joblib
import numpy as np
from cv2 import cv2

# keras.models.load_model()を使用してモデルを作成する
model = keras.models.load_model('catsdogs.h5')
# joblib.load()を使用して対応表をロード
animals = joblib.load('animal.pkl')

@hug.get("/")
def api_root():
    return "Development of Web API!"

# http://localhost:8000/catsdogs などにアクセスすると推定結果を返す
# フォームでPOSTを使用する
@hug.post("/catsdogs")
# 名前（2箇所）はHTMLのnameと合わせる
def api_catsdogs(name):
    img = np.asarray(bytearray(name),dtype=np.uint8)
    img2 = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img2,dsize=(150,150))
    img3 = img3[np.newaxis,...]
    img3 = img3/255.0
    # このimg3をpredictに渡すと予測できる
    result = model.predict(img3)
    arg = result[0][0].round().astype(int)
    # 最終的にはcatsかdogsを返す
    return animals[arg]
#api_keras.py
import hug
#その他必要なライブラリをインポートする
import keras
# モデルをロードする
# 変数=keras.models.load_model('ファイル名')を使用してモデルを作成する
model_iris1 = keras.models.load_model('iris1.h5')
# 回帰モデルは自身で作成する
model_iris2 = keras.models.load_model('iris2.h5')

import joblib
species = joblib.load('species.pkl')

@hug.get("/")
def api_root():
    return "Development of Web API!"

# http://localhost:8000/clf?x=0.1&x2=0.2 にアクセスすると分類問題の結果を返す
@hug.get("/iris1")
def api_iris1(x1, x2,x3,x4):
    x1, x2,x3,x4 = float(x1), float(x2),float(x3),float(x4)
    arg = model_iris1.predict([[x1, x2,x3,x4]]).argmax()
    return species[arg]

# http://localhost:8000/reg?x=0.1&x2=0.2 にアクセスすると回帰問題の結果を返す
@hug.get('/iris2')
def api_iris2(x1, x2,x3,x4):
    x1, x2,x3,x4 = float(x1), float(x2),float(x3),float(x4)
    arg = model_iris2.predict([[x1, x2,x3,x4]]).argmax()
    return species[arg]
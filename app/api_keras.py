#api_keras.py
import hug
#その他必要なライブラリをインポートする
import keras
# モデルをロードする
# 変数=keras.models.load_model('ファイル名')を使用してモデルを作成する
model_clf = keras.models.load_model('keras_clf.h5')
# 回帰モデルは自身で作成する
model_reg = keras.models.load_model('keras_reg.h5')


@hug.get("/")
def api_root():
    return "Development of Web API!"

# http://localhost:8000/clf?x=0.1&x2=0.2 にアクセスすると分類問題の結果を返す
@hug.get("/clf")
def api_clf(x1, x2):
    # x1,x2を浮動小数点数に変換
    x1, x2 = float(x1), float(x2)
    # model_clf.predictを使用して推定結果をreturnで直接返す
    return model_clf.predict([[x1, x2]])

# http://localhost:8000/reg?x=0.1&x2=0.2 にアクセスすると回帰問題の結果を返す
@hug.get('/reg')
def api_reg(x1, x2):
    # x1,x2を浮動小数点数に変換（作成する）
    x1, x2 = float(x1), float(x2)
    # model_reg.predictを使用して推定結果をreturnで直接返す（作成する）
    return model_reg.predict([[x1, x2]])
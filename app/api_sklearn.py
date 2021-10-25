
#api_sklearn.py
import hug
#その他必要なライブラリをインポートする
import sklearn
# joblib.load()を使用してモデルを作成する
import joblib
model = joblib.load('skl_xor.pkl')

@hug.get("/")
def api_root():
    return "Development of Web API!"

# http://localhost:8000/predict?x=0.1&x2=0.2 にアクセスすると推定結果を返す
# 直接指定またはフォームを使用する
@hug.get("/predict")
def api_predict(x1, x2):
    # x1,x2を浮動小数点数に変換
    x1,x2 = float(x1),float(x2)
    # model.predictを使用して推定結果をreturnで直接返す
    return model.predict([[x1,x2]])
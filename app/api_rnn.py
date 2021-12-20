#api_sklearn.py
import hug
#その他必要なライブラリをインポートする
import keras

# keras.models.load_model()を使用してモデルを作成する
model_rnn = keras.models.load_model('rnn.h5')
model_lstm = keras.models.load_model('lstm.h5')
model_gru = keras.models.load_model('gru.h5')

@hug.get("/")
def api_root():
    return "Development of Web API!"

# 直接指定またはフォームを使用する
@hug.get("/rnn")
def api_rnn(x_5,x_4,x_3,x_2,x_1):
    # 各変数を浮動小数点数に変換
    input = [float(i) for i in [x_5,x_4,x_3,x_2,x_1]]
    # 結果をreturnで返す。引数はモデル作成時のものと合わせる
    return model_rnn.predict([input])

@hug.get("/lstm")
def api_lstm(x_5,x_4,x_3,x_2,x_1):
    # 各変数を浮動小数点数に変換
    input = [float(i) for i in [x_5,x_4,x_3,x_2,x_1]]
    # 結果をreturnで返す。引数はモデル作成時のものと合わせる
    return model_lstm.predict([input])

@hug.get("/gru")
def api_gru(x_5,x_4,x_3,x_2,x_1):
    # 各変数を浮動小数点数に変換
    input = [float(i) for i in [x_5,x_4,x_3,x_2,x_1]]
    # 結果をreturnで返す。引数はモデル作成時のものと合わせる
    return model_gru.predict([input])

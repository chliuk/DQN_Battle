# 導入函式庫
import numpy as np
from keras.datasets import mnist
import keras
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import GRU, Activation, Dense
from keras.optimizers import adam_v2
from keras.utils.np_utils import to_categorical

import tensorflow as tf

# 打印 TensorFlow 可见的设备
devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU devices:", devices)


# 检查 GPU 是否可用
if tf.config.list_physical_devices():
    # 设置 TensorFlow 在 GPU 上运行
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
else:
    print("GPU not available. Using CPU.")



# 固定亂數種子，使每次執行產生的亂數都一樣
np.random.seed(1337)


# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 將 training 的 input 資料轉為3維，並 normalize 把顏色控制在 0 ~ 1 之間
X_train = X_train.reshape(-1, 28, 28) / 255.      
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes=10)


# 建立簡單的線性執行的模型
model = Sequential()
# 加 RNN 隱藏層(hidden layer)
model.add(GRU(
    # 如果後端使用tensorflow，batch_input_shape 的 batch_size 需設為 None.
    # 否則執行 model.evaluate() 會有錯誤產生.
    batch_input_shape=(None, 28, 28), 
    units= 50,
    unroll=True,
)) 
# 加 output 層
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
LR = 0.001          # Learning Rate
adam = adam_v2.Adam(LR)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 一批訓練多少張圖片
BATCH_SIZE = 50     
BATCH_INDEX = 0     
# 訓練模型 4001 次
for step in range(1, 4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    # 逐批訓練
    loss = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # 每 500 批，顯示測試的準確率
    if step % 500 == 0:
        # 模型評估
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], 
            verbose=False)
        print("test loss: {}  test accuracy: {}".format(loss,accuracy))
        

# 預測(prediction)
X = X_test[0:10,:]
predictions = model.predict(X)
# get prediction result
print(predictions)

# 模型結構存檔
from keras.models import model_from_json
json_string = model.to_json()
with open("GRU.config", "w") as text_file:
    text_file.write(json_string)

    
# 模型訓練結果存檔
model.save_weights("GRU.weight")

        
# HTSRC 高鐵訂票驗證碼

### 免責條款

此專案是個人學習如何使用 Deep Learning 中的 CNN，使用 Python 的 Keras、Tensorflow 進行實作，請勿使用於不法用途。若因使用該專案而大量訂票，相關的刑事、民事相關責任，請自行負責。

### 參考資料

其實基本上都是依照下面三個參考資料在實作，所以有疑問請在參考一下這些資料。

[gary9987/-Keras-Python3.6-captcha](https://github.com/gary9987/-Keras-TensorFlow-Python3.6-)

[[爬蟲實戰] 如何破解高鐵驗證碼 (1) - 去除圖片噪音點?](https://www.youtube.com/watch?v=6HGbKdB4kVY)

[[爬蟲實戰] 如何破解高鐵驗證碼 (2) - 使用迴歸方法去除多餘弧線?](https://www.youtube.com/watch?v=4DHcOPSfC4c)

### Dependencies

請先安裝相關的 python 套件

```sh
pip3 install -r requirements.txt
```

### 步驟

大致上分為四個步驟，以下會分步驟說明
- 爬蟲
- 預處理
- 標記圖片
- CNN深度學習

### 爬蟲

爬蟲請參考 `crawler.ipynb` 和編譯出來的 `crawler.py`。此程式使用 Selenium Chrome driver 去抓取高鐵螢幕截圖，再切割出驗證碼圖片存入至 captcha 目錄下。

需要注意的是，因為我使用 Macbook Pro 的 Retina 螢幕，使用螢幕截圖時，解析度會自動變為2倍，所以中間有一段程式在處理這個 ratio，不過最後都存成 140 x 48 的圖片。

檔案列表：

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | crawler.ipynb | crawler.py |

### 預處理

圖片預處理就參考[參考資料](#參考資料)的 youtube 教學影片，比較麻煩的是處理上方一條線。這部分我認為預處理沒有作的比參考資料1弄的漂亮，主要是因為其實高鐵的驗證碼圖片大小不是固定的，若是刪除此弧線若是可以根據圖片的高度，這樣效果會更好，但是我基本上分為三個步驟處理圖片，沒有把步驟1和步驟2合起來處理。

檔案列表：

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | preprocess.ipynb |  |
| 2 | preprocess-batch.ipynb | preprocess-batch.py |

### 標記圖片

標記圖片花了很多時間，所以中間衍生我弄了另外一個專案 [label_captcha_tool](https://github.com/maxmilian/label_captcha_tool)。主要是因為找了一些標記工具，發現有些是 windows 的，而或者是安裝有點麻煩，不如我就自己寫了一套網頁的版本，當然好處就是可以跨平台，也就是一個 html 而已，也不用安裝，算是大大節省我的時間。

這邊就標記檔，存成為 csv (`label.csv`)，每一個圖片一行，之後要丟入 CNN 當作 label 的訓練資料。

### CNN深度學習

CNN部分就直接使用參考資料1，這部分優化比較少

| # | ipython notebook 檔 | python檔 |
|---|---|---|
| 1 | cnn.ipynb | |

這邊就標記檔，存成為 csv，每一個圖片一行，之後要丟入 CNN 當作 label 的訓練資料。

model.summary()

```sh
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 48, 140, 3)   0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 48, 140, 32)  896         input_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 46, 138, 32)  9248        conv2d_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 23, 69, 32)   0           conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 23, 69, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 21, 67, 64)   36928       conv2d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 10, 33, 64)   0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 10, 33, 128)  73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 31, 128)   147584      conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 8, 31, 128)   32          conv2d_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 4, 15, 128)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 15, 256)   295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 15, 256)   590080      conv2d_7[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 7, 256)    0           conv2d_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 2, 7, 512)    1180160     max_pooling2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 2, 7, 512)    8           conv2d_9[0][0]
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 1, 3, 512)    0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1536)         0           max_pooling2d_5[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1536)         0           flatten_1[0][0]
__________________________________________________________________________________________________
digit1 (Dense)                  (None, 19)           29203       dropout_1[0][0]
__________________________________________________________________________________________________
digit2 (Dense)                  (None, 19)           29203       dropout_1[0][0]
__________________________________________________________________________________________________
digit3 (Dense)                  (None, 19)           29203       dropout_1[0][0]
__________________________________________________________________________________________________
digit4 (Dense)                  (None, 19)           29203       dropout_1[0][0]
==================================================================================================
Total params: 2,469,268
Trainable params: 2,469,248
Non-trainable params: 20
```

model.fit log

```sh
Train on 4000 samples, validate on 1000 samples

Epoch 1/30
4000/4000 [==============================] - 126s 32ms/step - loss: 16.0540 - digit1_loss: 4.0371 - digit2_loss: 3.9618 - digit3_loss: 4.0593 - digit4_loss: 3.9958 - digit1_acc: 0.0552 - digit2_acc: 0.0515 - digit3_acc: 0.0530 - digit4_acc: 0.0575 - val_loss: 12.2010 - val_digit1_loss: 2.9647 - val_digit2_loss: 3.0668 - val_digit3_loss: 3.0576 - val_digit4_loss: 3.1119 - val_digit1_acc: 0.0330 - val_digit2_acc: 0.0540 - val_digit3_acc: 0.0590 - val_digit4_acc: 0.0510

Epoch 00001: saving model to model/01-16.05-12.20.hdf5
Epoch 2/30
4000/4000 [==============================] - 121s 30ms/step - loss: 13.6571 - digit1_loss: 3.3964 - digit2_loss: 3.4085 - digit3_loss: 3.3932 - digit4_loss: 3.4590 - digit1_acc: 0.0580 - digit2_acc: 0.0575 - digit3_acc: 0.0557 - digit4_acc: 0.0517 - val_loss: 11.9186 - val_digit1_loss: 2.9670 - val_digit2_loss: 2.9621 - val_digit3_loss: 2.9997 - val_digit4_loss: 2.9898 - val_digit1_acc: 0.0640 - val_digit2_acc: 0.0610 - val_digit3_acc: 0.1030 - val_digit4_acc: 0.0520

Epoch 00002: saving model to model/02-13.66-11.92.hdf5
Epoch 3/30
4000/4000 [==============================] - 122s 31ms/step - loss: 12.2014 - digit1_loss: 3.0349 - digit2_loss: 3.0423 - digit3_loss: 3.0436 - digit4_loss: 3.0806 - digit1_acc: 0.0605 - digit2_acc: 0.0570 - digit3_acc: 0.0612 - digit4_acc: 0.0590 - val_loss: 11.8083 - val_digit1_loss: 2.9481 - val_digit2_loss: 2.9509 - val_digit3_loss: 2.9422 - val_digit4_loss: 2.9670 - val_digit1_acc: 0.0760 - val_digit2_acc: 0.0480 - val_digit3_acc: 0.0600 - val_digit4_acc: 0.0590

Epoch 00003: saving model to model/03-12.20-11.81.hdf5
Epoch 4/30
4000/4000 [==============================] - 121s 30ms/step - loss: 11.8352 - digit1_loss: 2.9504 - digit2_loss: 2.9566 - digit3_loss: 2.9638 - digit4_loss: 2.9644 - digit1_acc: 0.0552 - digit2_acc: 0.0532 - digit3_acc: 0.0490 - digit4_acc: 0.0562 - val_loss: 11.7646 - val_digit1_loss: 2.9332 - val_digit2_loss: 2.9459 - val_digit3_loss: 2.9347 - val_digit4_loss: 2.9507 - val_digit1_acc: 0.0890 - val_digit2_acc: 0.0750 - val_digit3_acc: 0.0720 - val_digit4_acc: 0.0530

Epoch 00004: saving model to model/04-11.84-11.76.hdf5
Epoch 5/30
4000/4000 [==============================] - 121s 30ms/step - loss: 11.7993 - digit1_loss: 2.9391 - digit2_loss: 2.9512 - digit3_loss: 2.9531 - digit4_loss: 2.9558 - digit1_acc: 0.0640 - digit2_acc: 0.0587 - digit3_acc: 0.0545 - digit4_acc: 0.0552 - val_loss: 11.7560 - val_digit1_loss: 2.9320 - val_digit2_loss: 2.9450 - val_digit3_loss: 2.9331 - val_digit4_loss: 2.9458 - val_digit1_acc: 0.0560 - val_digit2_acc: 0.0650 - val_digit3_acc: 0.0700 - val_digit4_acc: 0.0460

Epoch 00005: saving model to model/05-11.80-11.76.hdf5
Epoch 6/30
4000/4000 [==============================] - 122s 31ms/step - loss: 11.7456 - digit1_loss: 2.9215 - digit2_loss: 2.9310 - digit3_loss: 2.9360 - digit4_loss: 2.9572 - digit1_acc: 0.0602 - digit2_acc: 0.0717 - digit3_acc: 0.0657 - digit4_acc: 0.0510 - val_loss: 11.7722 - val_digit1_loss: 2.9454 - val_digit2_loss: 2.9480 - val_digit3_loss: 2.9374 - val_digit4_loss: 2.9413 - val_digit1_acc: 0.0610 - val_digit2_acc: 0.0590 - val_digit3_acc: 0.0690 - val_digit4_acc: 0.0590

Epoch 00006: saving model to model/06-11.75-11.77.hdf5
Epoch 7/30
4000/4000 [==============================] - 121s 30ms/step - loss: 11.4043 - digit1_loss: 2.8127 - digit2_loss: 2.8095 - digit3_loss: 2.8426 - digit4_loss: 2.9395 - digit1_acc: 0.1035 - digit2_acc: 0.1175 - digit3_acc: 0.0927 - digit4_acc: 0.0610 - val_loss: 11.4247 - val_digit1_loss: 2.8225 - val_digit2_loss: 2.8337 - val_digit3_loss: 2.8499 - val_digit4_loss: 2.9186 - val_digit1_acc: 0.1530 - val_digit2_acc: 0.1300 - val_digit3_acc: 0.1210 - val_digit4_acc: 0.0850

Epoch 00007: saving model to model/07-11.40-11.42.hdf5
Epoch 8/30
4000/4000 [==============================] - 118s 30ms/step - loss: 11.1589 - digit1_loss: 2.7326 - digit2_loss: 2.7388 - digit3_loss: 2.7704 - digit4_loss: 2.9171 - digit1_acc: 0.1293 - digit2_acc: 0.1262 - digit3_acc: 0.1120 - digit4_acc: 0.0807 - val_loss: 11.8702 - val_digit1_loss: 3.0091 - val_digit2_loss: 3.0127 - val_digit3_loss: 2.9173 - val_digit4_loss: 2.9311 - val_digit1_acc: 0.0600 - val_digit2_acc: 0.0610 - val_digit3_acc: 0.0690 - val_digit4_acc: 0.0750

Epoch 00008: saving model to model/08-11.16-11.87.hdf5
Epoch 9/30
4000/4000 [==============================] - 114s 28ms/step - loss: 10.3355 - digit1_loss: 2.4393 - digit2_loss: 2.5085 - digit3_loss: 2.5554 - digit4_loss: 2.8323 - digit1_acc: 0.2117 - digit2_acc: 0.1700 - digit3_acc: 0.1627 - digit4_acc: 0.1035 - val_loss: 11.5317 - val_digit1_loss: 2.9173 - val_digit2_loss: 2.8979 - val_digit3_loss: 2.8238 - val_digit4_loss: 2.8928 - val_digit1_acc: 0.1090 - val_digit2_acc: 0.0710 - val_digit3_acc: 0.0810 - val_digit4_acc: 0.0840

Epoch 00009: saving model to model/09-10.34-11.53.hdf5
Epoch 10/30
4000/4000 [==============================] - 113s 28ms/step - loss: 9.5355 - digit1_loss: 2.0846 - digit2_loss: 2.3134 - digit3_loss: 2.3553 - digit4_loss: 2.7823 - digit1_acc: 0.3115 - digit2_acc: 0.2440 - digit3_acc: 0.2330 - digit4_acc: 0.1227 - val_loss: 10.8557 - val_digit1_loss: 2.6697 - val_digit2_loss: 2.6968 - val_digit3_loss: 2.6556 - val_digit4_loss: 2.8335 - val_digit1_acc: 0.1600 - val_digit2_acc: 0.1640 - val_digit3_acc: 0.1800 - val_digit4_acc: 0.1300

Epoch 00010: saving model to model/10-9.54-10.86.hdf5
Epoch 11/30
4000/4000 [==============================] - 113s 28ms/step - loss: 8.7521 - digit1_loss: 1.7619 - digit2_loss: 2.1221 - digit3_loss: 2.1831 - digit4_loss: 2.6849 - digit1_acc: 0.4043 - digit2_acc: 0.2925 - digit3_acc: 0.2700 - digit4_acc: 0.1475 - val_loss: 10.2410 - val_digit1_loss: 2.3964 - val_digit2_loss: 2.5543 - val_digit3_loss: 2.5206 - val_digit4_loss: 2.7696 - val_digit1_acc: 0.2620 - val_digit2_acc: 0.2230 - val_digit3_acc: 0.2260 - val_digit4_acc: 0.1710

Epoch 00011: saving model to model/11-8.75-10.24.hdf5
Epoch 12/30
4000/4000 [==============================] - 114s 29ms/step - loss: 7.7067 - digit1_loss: 1.3698 - digit2_loss: 1.8615 - digit3_loss: 1.9268 - digit4_loss: 2.5485 - digit1_acc: 0.5265 - digit2_acc: 0.3683 - digit3_acc: 0.3503 - digit4_acc: 0.1867 - val_loss: 9.7986 - val_digit1_loss: 2.2391 - val_digit2_loss: 2.4483 - val_digit3_loss: 2.4261 - val_digit4_loss: 2.6851 - val_digit1_acc: 0.2580 - val_digit2_acc: 0.2430 - val_digit3_acc: 0.2580 - val_digit4_acc: 0.1680

Epoch 00012: saving model to model/12-7.71-9.80.hdf5
Epoch 13/30
4000/4000 [==============================] - 114s 28ms/step - loss: 6.6032 - digit1_loss: 1.0812 - digit2_loss: 1.6077 - digit3_loss: 1.6616 - digit4_loss: 2.2526 - digit1_acc: 0.6160 - digit2_acc: 0.4503 - digit3_acc: 0.4362 - digit4_acc: 0.2765 - val_loss: 8.7999 - val_digit1_loss: 1.8599 - val_digit2_loss: 2.2249 - val_digit3_loss: 2.2121 - val_digit4_loss: 2.5030 - val_digit1_acc: 0.4410 - val_digit2_acc: 0.2740 - val_digit3_acc: 0.3120 - val_digit4_acc: 0.2710

Epoch 00013: saving model to model/13-6.60-8.80.hdf5
Epoch 14/30
4000/4000 [==============================] - 114s 29ms/step - loss: 5.4419 - digit1_loss: 0.8701 - digit2_loss: 1.3448 - digit3_loss: 1.3916 - digit4_loss: 1.8355 - digit1_acc: 0.7070 - digit2_acc: 0.5397 - digit3_acc: 0.5300 - digit4_acc: 0.3962 - val_loss: 8.7488 - val_digit1_loss: 2.0633 - val_digit2_loss: 2.1930 - val_digit3_loss: 2.1801 - val_digit4_loss: 2.3125 - val_digit1_acc: 0.3980 - val_digit2_acc: 0.3090 - val_digit3_acc: 0.3180 - val_digit4_acc: 0.2770

Epoch 00014: saving model to model/14-5.44-8.75.hdf5
Epoch 15/30
4000/4000 [==============================] - 114s 28ms/step - loss: 4.2047 - digit1_loss: 0.6879 - digit2_loss: 1.0596 - digit3_loss: 1.0408 - digit4_loss: 1.4164 - digit1_acc: 0.7690 - digit2_acc: 0.6370 - digit3_acc: 0.6415 - digit4_acc: 0.5135 - val_loss: 7.1819 - val_digit1_loss: 1.6352 - val_digit2_loss: 1.8690 - val_digit3_loss: 1.8067 - val_digit4_loss: 1.8710 - val_digit1_acc: 0.5080 - val_digit2_acc: 0.4470 - val_digit3_acc: 0.4680 - val_digit4_acc: 0.5110

Epoch 00015: saving model to model/15-4.20-7.18.hdf5
Epoch 16/30
4000/4000 [==============================] - 115s 29ms/step - loss: 3.1712 - digit1_loss: 0.5458 - digit2_loss: 0.7972 - digit3_loss: 0.7973 - digit4_loss: 1.0309 - digit1_acc: 0.8240 - digit2_acc: 0.7323 - digit3_acc: 0.7272 - digit4_acc: 0.6578 - val_loss: 5.3002 - val_digit1_loss: 1.2007 - val_digit2_loss: 1.3546 - val_digit3_loss: 1.3279 - val_digit4_loss: 1.4170 - val_digit1_acc: 0.6370 - val_digit2_acc: 0.6050 - val_digit3_acc: 0.6000 - val_digit4_acc: 0.6730

Epoch 00016: saving model to model/16-3.17-5.30.hdf5
Epoch 17/30
4000/4000 [==============================] - 114s 28ms/step - loss: 2.4053 - digit1_loss: 0.4284 - digit2_loss: 0.6214 - digit3_loss: 0.6057 - digit4_loss: 0.7498 - digit1_acc: 0.8675 - digit2_acc: 0.7997 - digit3_acc: 0.8102 - digit4_acc: 0.7528 - val_loss: 5.8780 - val_digit1_loss: 1.5331 - val_digit2_loss: 1.6148 - val_digit3_loss: 1.4048 - val_digit4_loss: 1.3253 - val_digit1_acc: 0.5270 - val_digit2_acc: 0.4860 - val_digit3_acc: 0.5620 - val_digit4_acc: 0.6130

Epoch 00017: saving model to model/17-2.41-5.88.hdf5
Epoch 18/30
4000/4000 [==============================] - 113s 28ms/step - loss: 1.8415 - digit1_loss: 0.3371 - digit2_loss: 0.4753 - digit3_loss: 0.4630 - digit4_loss: 0.5661 - digit1_acc: 0.8962 - digit2_acc: 0.8530 - digit3_acc: 0.8472 - digit4_acc: 0.8202 - val_loss: 2.2810 - val_digit1_loss: 0.5258 - val_digit2_loss: 0.6435 - val_digit3_loss: 0.5563 - val_digit4_loss: 0.5554 - val_digit1_acc: 0.8430 - val_digit2_acc: 0.8150 - val_digit3_acc: 0.8410 - val_digit4_acc: 0.8770

Epoch 00018: saving model to model/18-1.84-2.28.hdf5
Epoch 19/30
4000/4000 [==============================] - 114s 29ms/step - loss: 1.3941 - digit1_loss: 0.2920 - digit2_loss: 0.3748 - digit3_loss: 0.3493 - digit4_loss: 0.3781 - digit1_acc: 0.9165 - digit2_acc: 0.8820 - digit3_acc: 0.8937 - digit4_acc: 0.8880 - val_loss: 2.5405 - val_digit1_loss: 0.6710 - val_digit2_loss: 0.7068 - val_digit3_loss: 0.6372 - val_digit4_loss: 0.5254 - val_digit1_acc: 0.7660 - val_digit2_acc: 0.7680 - val_digit3_acc: 0.7900 - val_digit4_acc: 0.8520

Epoch 00019: saving model to model/19-1.39-2.54.hdf5
Epoch 20/30
4000/4000 [==============================] - 114s 28ms/step - loss: 1.1144 - digit1_loss: 0.2260 - digit2_loss: 0.3042 - digit3_loss: 0.2801 - digit4_loss: 0.3041 - digit1_acc: 0.9415 - digit2_acc: 0.9115 - digit3_acc: 0.9120 - digit4_acc: 0.9115 - val_loss: 1.7021 - val_digit1_loss: 0.4025 - val_digit2_loss: 0.5123 - val_digit3_loss: 0.3821 - val_digit4_loss: 0.4053 - val_digit1_acc: 0.8710 - val_digit2_acc: 0.8450 - val_digit3_acc: 0.8810 - val_digit4_acc: 0.8760

Epoch 00020: saving model to model/20-1.11-1.70.hdf5
Epoch 21/30
4000/4000 [==============================] - 114s 28ms/step - loss: 0.8938 - digit1_loss: 0.1923 - digit2_loss: 0.2484 - digit3_loss: 0.2102 - digit4_loss: 0.2429 - digit1_acc: 0.9460 - digit2_acc: 0.9235 - digit3_acc: 0.9392 - digit4_acc: 0.9297 - val_loss: 0.6163 - val_digit1_loss: 0.1318 - val_digit2_loss: 0.1763 - val_digit3_loss: 0.1554 - val_digit4_loss: 0.1528 - val_digit1_acc: 0.9650 - val_digit2_acc: 0.9530 - val_digit3_acc: 0.9570 - val_digit4_acc: 0.9680

Epoch 00021: saving model to model/21-0.89-0.62.hdf5
Epoch 22/30
4000/4000 [==============================] - 114s 28ms/step - loss: 0.7123 - digit1_loss: 0.1675 - digit2_loss: 0.1975 - digit3_loss: 0.1650 - digit4_loss: 0.1823 - digit1_acc: 0.9540 - digit2_acc: 0.9445 - digit3_acc: 0.9557 - digit4_acc: 0.9495 - val_loss: 1.2015 - val_digit1_loss: 0.3690 - val_digit2_loss: 0.3247 - val_digit3_loss: 0.2916 - val_digit4_loss: 0.2162 - val_digit1_acc: 0.8760 - val_digit2_acc: 0.8960 - val_digit3_acc: 0.9070 - val_digit4_acc: 0.9370

Epoch 00022: saving model to model/22-0.71-1.20.hdf5
Epoch 23/30
4000/4000 [==============================] - 113s 28ms/step - loss: 0.6192 - digit1_loss: 0.1450 - digit2_loss: 0.1707 - digit3_loss: 0.1424 - digit4_loss: 0.1611 - digit1_acc: 0.9570 - digit2_acc: 0.9500 - digit3_acc: 0.9612 - digit4_acc: 0.9562 - val_loss: 0.7501 - val_digit1_loss: 0.2225 - val_digit2_loss: 0.2332 - val_digit3_loss: 0.1621 - val_digit4_loss: 0.1323 - val_digit1_acc: 0.9240 - val_digit2_acc: 0.9210 - val_digit3_acc: 0.9510 - val_digit4_acc: 0.9680

Epoch 00023: saving model to model/23-0.62-0.75.hdf5
Epoch 24/30
4000/4000 [==============================] - 117s 29ms/step - loss: 0.5221 - digit1_loss: 0.1337 - digit2_loss: 0.1410 - digit3_loss: 0.1055 - digit4_loss: 0.1419 - digit1_acc: 0.9632 - digit2_acc: 0.9585 - digit3_acc: 0.9720 - digit4_acc: 0.9668 - val_loss: 0.3654 - val_digit1_loss: 0.0922 - val_digit2_loss: 0.0996 - val_digit3_loss: 0.1112 - val_digit4_loss: 0.0624 - val_digit1_acc: 0.9730 - val_digit2_acc: 0.9770 - val_digit3_acc: 0.9670 - val_digit4_acc: 0.9850

Epoch 00024: saving model to model/24-0.52-0.37.hdf5
Epoch 25/30
4000/4000 [==============================] - 114s 28ms/step - loss: 0.4145 - digit1_loss: 0.1008 - digit2_loss: 0.1110 - digit3_loss: 0.0954 - digit4_loss: 0.1074 - digit1_acc: 0.9717 - digit2_acc: 0.9640 - digit3_acc: 0.9743 - digit4_acc: 0.9755 - val_loss: 0.3294 - val_digit1_loss: 0.0892 - val_digit2_loss: 0.0964 - val_digit3_loss: 0.0870 - val_digit4_loss: 0.0567 - val_digit1_acc: 0.9720 - val_digit2_acc: 0.9720 - val_digit3_acc: 0.9770 - val_digit4_acc: 0.9870

Epoch 00025: saving model to model/25-0.41-0.33.hdf5
Epoch 26/30
4000/4000 [==============================] - 116s 29ms/step - loss: 0.3468 - digit1_loss: 0.0857 - digit2_loss: 0.0908 - digit3_loss: 0.0757 - digit4_loss: 0.0946 - digit1_acc: 0.9750 - digit2_acc: 0.9723 - digit3_acc: 0.9798 - digit4_acc: 0.9763 - val_loss: 0.3806 - val_digit1_loss: 0.1024 - val_digit2_loss: 0.1018 - val_digit3_loss: 0.1081 - val_digit4_loss: 0.0682 - val_digit1_acc: 0.9660 - val_digit2_acc: 0.9760 - val_digit3_acc: 0.9700 - val_digit4_acc: 0.9840

Epoch 00026: saving model to model/26-0.35-0.38.hdf5
Epoch 27/30
4000/4000 [==============================] - 114s 29ms/step - loss: 0.3223 - digit1_loss: 0.0731 - digit2_loss: 0.0810 - digit3_loss: 0.0756 - digit4_loss: 0.0927 - digit1_acc: 0.9790 - digit2_acc: 0.9745 - digit3_acc: 0.9788 - digit4_acc: 0.9790 - val_loss: 0.3812 - val_digit1_loss: 0.0831 - val_digit2_loss: 0.1155 - val_digit3_loss: 0.0933 - val_digit4_loss: 0.0892 - val_digit1_acc: 0.9770 - val_digit2_acc: 0.9660 - val_digit3_acc: 0.9690 - val_digit4_acc: 0.9680

Epoch 00027: saving model to model/27-0.32-0.38.hdf5
Epoch 28/30
4000/4000 [==============================] - 114s 29ms/step - loss: 0.2652 - digit1_loss: 0.0594 - digit2_loss: 0.0678 - digit3_loss: 0.0653 - digit4_loss: 0.0727 - digit1_acc: 0.9808 - digit2_acc: 0.9803 - digit3_acc: 0.9823 - digit4_acc: 0.9818 - val_loss: 0.2595 - val_digit1_loss: 0.0624 - val_digit2_loss: 0.0946 - val_digit3_loss: 0.0656 - val_digit4_loss: 0.0368 - val_digit1_acc: 0.9820 - val_digit2_acc: 0.9740 - val_digit3_acc: 0.9830 - val_digit4_acc: 0.9920

Epoch 00028: saving model to model/28-0.27-0.26.hdf5
Epoch 29/30
4000/4000 [==============================] - 114s 29ms/step - loss: 0.2314 - digit1_loss: 0.0593 - digit2_loss: 0.0581 - digit3_loss: 0.0512 - digit4_loss: 0.0629 - digit1_acc: 0.9845 - digit2_acc: 0.9800 - digit3_acc: 0.9858 - digit4_acc: 0.9838 - val_loss: 0.4200 - val_digit1_loss: 0.1045 - val_digit2_loss: 0.1370 - val_digit3_loss: 0.1012 - val_digit4_loss: 0.0773 - val_digit1_acc: 0.9640 - val_digit2_acc: 0.9590 - val_digit3_acc: 0.9660 - val_digit4_acc: 0.9750

Epoch 00029: saving model to model/29-0.23-0.42.hdf5
Epoch 30/30
4000/4000 [==============================] - 114s 29ms/step - loss: 0.2129 - digit1_loss: 0.0534 - digit2_loss: 0.0542 - digit3_loss: 0.0497 - digit4_loss: 0.0556 - digit1_acc: 0.9840 - digit2_acc: 0.9818 - digit3_acc: 0.9843 - digit4_acc: 0.9873 - val_loss: 0.2538 - val_digit1_loss: 0.0476 - val_digit2_loss: 0.0837 - val_digit3_loss: 0.0828 - val_digit4_loss: 0.0397 - val_digit1_acc: 0.9850 - val_digit2_acc: 0.9750 - val_digit3_acc: 0.9770 - val_digit4_acc: 0.9910

Epoch 00030: saving model to model/30-0.21-0.25.hdf5
```

#### 訓練 log
![Train History](train_history.png)

# 其他

把 Jupyter Notebook 轉為 python script

```
# 爬蟲
jupyter nbconvert --to script crawler.ipynb

# 影像預處理
jupyter nbconvert --to script preprocessBatch.ipynb
```

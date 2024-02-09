# 필요한 라이브러리와 모듈 임포트
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# 설정값과 관련한 변수
path = './dataset/'
dataset = 'ratings.csv'
layers = [64, 32, 16, 8]  # 각 층의 노드 수
epochs = 10
batch_size = 32
regs = [0, 0, 0, 0]  # 정규화 계수
lr = 0.001  # 학습률
learner = 'adam'
out = 1
patience = 10

# 데이터 로딩 및 전처리
loader = DataLoader(path + dataset)
X_train, labels = loader.generate_trainset()
X_test, test_labels = loader.generate_testset()

# 모델 구성
num_users, num_items = loader.num_users, loader.num_items

user_input = keras.layers.Input(shape=(1,), dtype='int32')
item_input = keras.layers.Input(shape=(1,), dtype='int32')

# 첫 번째 임베딩 레이어의 크기는 layers[0]의 절반으로 설정
user_embedding = keras.layers.Embedding(input_dim=num_users, output_dim=int(layers[0]/2), embeddings_regularizer=keras.regularizers.l2(regs[0]), name="user_embedding")(user_input)
item_embedding = keras.layers.Embedding(input_dim=num_items, output_dim=int(layers[0]/2), embeddings_regularizer=keras.regularizers.l2(regs[0]), name='item_embedding')(item_input)

user_latent = keras.layers.Flatten()(user_embedding)
item_latent = keras.layers.Flatten()(item_embedding)

vector = keras.layers.concatenate([user_latent, item_latent])

# 숨겨진 층 추가
for index, layer_size in enumerate(layers):
    vector = keras.layers.Dense(layer_size, kernel_regularizer=keras.regularizers.l2(regs[index]), activation='relu', name=f'layer{index}')(vector)

output = keras.layers.Dense(1, kernel_initializer=keras.initializers.lecun_uniform(), name='output')(vector)

model = keras.Model(inputs=[user_input, item_input], outputs=[output])

# 모델 컴파일
if learner.lower() == "adam":
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')

# 모델 훈련 및 평가
early_stop_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
model_out_file = 'Pretrain/MLP_%s.h5' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

history = model.fit([X_train[:, 0], X_train[:, 1]], labels, epochs=epochs, batch_size=batch_size, validation_data=([X_test[:, 0], X_test[:, 1]], test_labels), callbacks=[early_stop_cb, model_check_cb] if out else [early_stop_cb])

# 훈련 과정 시각화
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

# 테스트 데이터에 대한 예측
test_sample = X_test[:10]
predictions = model.predict([test_sample[:, 0], test_sample[:, 1]])
print(predictions)
print(test_labels[:10]) 

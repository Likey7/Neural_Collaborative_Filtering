from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import DataLoader

# 설정값과 관련한 변수
path = './dataset/'
dataset = 'ratings.csv'
layers = [64, 32, 16, 8]
num_factors = 8
epochs = 10
batch_size = 32
gmf_regs = 0.0
mlp_regs = [0.0, 0.0, 0.0, 0.0]
lr = 0.001
learner = 'adam'
out = 1
patience = 10
pretrain_gmf = ''
pretrain_mlp = ''
alpha = 0.5

# 데이터셋 로드
loader = DataLoader(path + dataset)
X_train, labels = loader.generate_trainset()
X_test, test_labels = loader.generate_testset()

# 모델 구성
num_users, num_items = loader.num_users, loader.num_items

# 입력 레이어
user_input = keras.layers.Input(shape=(1,), dtype='int32')
item_input = keras.layers.Input(shape=(1,), dtype='int32')

# GMF 임베딩 레이어
user_embedding_gmf = keras.layers.Embedding(input_dim=num_users, output_dim=num_factors, embeddings_regularizer=keras.regularizers.l2(gmf_regs), name='user_embedding_gmf')(user_input)
item_embedding_gmf = keras.layers.Embedding(input_dim=num_items, output_dim=num_factors, embeddings_regularizer=keras.regularizers.l2(gmf_regs), name='item_embedding_gmf')(item_input)
user_latent_gmf = keras.layers.Flatten()(user_embedding_gmf)
item_latent_gmf = keras.layers.Flatten()(item_embedding_gmf)
result_gmf = keras.layers.Multiply()([user_latent_gmf, item_latent_gmf])

# MLP 임베딩 레이어
user_embedding_mlp = keras.layers.Embedding(input_dim=num_users, output_dim=int(layers[0]/2), embeddings_regularizer=keras.regularizers.l2(mlp_regs[0]), name='user_embedding_mlp')(user_input)
item_embedding_mlp = keras.layers.Embedding(input_dim=num_items, output_dim=int(layers[0]/2), embeddings_regularizer=keras.regularizers.l2(mlp_regs[0]), name='item_embedding_mlp')(item_input)
user_latent_mlp = keras.layers.Flatten()(user_embedding_mlp)
item_latent_mlp = keras.layers.Flatten()(item_embedding_mlp)
result_mlp = keras.layers.concatenate([user_latent_mlp, item_latent_mlp])

# MLP 히든 레이어
for index, layer_size in enumerate(layers):
    result_mlp = keras.layers.Dense(layer_size, kernel_regularizer=keras.regularizers.l2(mlp_regs[index]), activation='relu', name=f'layer{index}')(result_mlp)

# GMF와 MLP 결과를 결합
concat = keras.layers.concatenate([result_gmf, result_mlp])

# 최종 출력 레이어
output = keras.layers.Dense(1, kernel_initializer=keras.initializers.lecun_uniform(), name='output')(concat)

# 모델 생성
model = keras.Model(inputs=[user_input, item_input], outputs=output)

# 모델 컴파일
if learner == 'adam':
    optimizer = keras.optimizers.Adam(learning_rate=lr)
else:
    optimizer = keras.optimizers.SGD(learning_rate=lr) # 기본값을 adam으로 설정함, 다른 옵션 필요시 추가 구현하면됨.
model.compile(optimizer=optimizer, loss='mse')

# 모델 훈련
early_stop_cb = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
model_out_file = 'Pretrain/NeuralMF_%s.h5' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

history = model.fit([X_train[:, 0], X_train[:, 1]], labels, epochs=epochs, batch_size=batch_size, validation_data=([X_test[:, 0], X_test[:, 1]], test_labels), callbacks=[early_stop_cb, model_check_cb])

# 훈련 과정 시각화
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

# 테스트 데이터에 대한 예측
test_sample = X_test[:10]
predictions = model.predict([test_sample[:, 0], test_sample[:, 1]])
print(predictions)
print(test_labels[:10])

# 필요한 라이브러리와 모듈 임포트
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# 설정값과 관련한 변수
config = {
    'path': './dataset/',
    'dataset': 'ratings.csv',
    'epochs': 10,
    'batch_size': 32,
    'latent_features': 8,
    'regularization': 0.0,
    'learning_rate': 0.001,
    'optimizer_type': 'adam',
    'output': True,
    'patience': 10
}

# 데이터셋 로드 및 전처리
loader = DataLoader(config['path'] + config['dataset'])
X_train, labels = loader.generate_trainset()
X_test, test_labels = loader.generate_testset()


# 모델 정의
num_users, num_items = loader.num_users, loader.num_items
latent_features, regularization = config['latent_features'], config['regularization']

user_input = keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
item_input = keras.layers.Input(shape=(1,), dtype='int32', name='item_input')

user_embedding = keras.layers.Embedding(input_dim=num_users, output_dim=latent_features,
                                        embeddings_regularizer=keras.regularizers.l2(regularization),
                                        name='user_embedding')(user_input)
item_embedding = keras.layers.Embedding(input_dim=num_items, output_dim=latent_features,
                                        embeddings_regularizer=keras.regularizers.l2(regularization),
                                        name='item_embedding')(item_input)

user_latent = keras.layers.Flatten()(user_embedding)
item_latent = keras.layers.Flatten()(item_embedding)

interaction = keras.layers.Multiply()([user_latent, item_latent])

output = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='output')(interaction)

model = keras.Model(inputs=[user_input, item_input], outputs=output)

def configure_optimizer(learning_rate, optimizer_type='adam'):
    """
    모델 컴파일을 위한 최적화 함수

    매개변수:
    - learning_rate: 최적화 함수의 학습률
    - optimizer_type: 사용할 최적화 함수의 타입('adam', 'sgd' 등)

    반환값:
    - keras.optimizers의 최적화 함수 인스턴스
    """
    optimizers = {
        'adagrad': keras.optimizers.Adagrad(learning_rate=learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
        'adam': keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': keras.optimizers.SGD(learning_rate=learning_rate),
    }
    return optimizers[optimizer_type.lower()]

# 모델 컴파일(위에서 만들어놓은 configure_optimizer 기능 사용함!)
optimizer = configure_optimizer(config['learning_rate'], config['optimizer_type'])
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# 콜백 정의
early_stop_cb = keras.callbacks.EarlyStopping(patience=config['patience'], restore_best_weights=True)
model_out_file = f'Pretrain/GMF_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.h5'
model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)


# 모델 훈련
history = model.fit([X_train[:, 0], X_train[:, 1]], labels,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    validation_data=([X_test[:, 0], X_test[:, 1]], test_labels),
                    callbacks=[early_stop_cb, model_check_cb] if config['output'] else [early_stop_cb])

# 훈련 및 검증 손실 추출
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 손실 그래프 시각화
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 테스트 데이터에 대한 예측
test_sample = X_test[:10]
predictions = model.predict([test_sample[:, 0], test_sample[:, 1]])
print(predictions)
print(test_labels[:10])
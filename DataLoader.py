import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    영화 평점 데이터셋을 로드하고, 훈련 및 테스트 세트로 분리하는 클래스
    사용자와 영화 ID를 숫자 인덱스로 매핑하여 데이터 전처리
    """

    def __init__(self, data_path):
        """
        DataLoader 클래스 생성자

        Args:
            data_path (str): 평점 데이터 파일의 경로
        """
        self.data_path = data_path
        # 데이터 파일을 읽어 DataFrame을 생성
        ratings_df = pd.read_csv(os.path.join(self.data_path), encoding='utf-8')
        
        # 데이터를 훈련 세트와 테스트 세트로 분리
        self.train_df, self.test_df = train_test_split(ratings_df, test_size=0.2, random_state=42, shuffle=True)
        
        # 사용자 데이터 처리
        self.users = self.train_df["userId"].unique()
        self.num_users = len(self.users)
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}
        
        # 영화 데이터 처리
        self.movies = self.train_df["movieId"].unique()
        self.num_items = len(self.movies)
        self.movies_to_index = {movie: idx for idx, movie in enumerate(self.movies)}

        # 테스트 데이터셋에서 훈련 데이터셋에 없는 사용자나 영화를 제외
        self.test_df = self.test_df[self.test_df["userId"].isin(self.users) & self.test_df["movieId"].isin(self.movies)]

    def generate_trainset(self):
        """
        훈련 데이터셋을 생성

        Returns:
            tuple: 사용자와 영화 인덱스로 구성된 배열, 평점 배열
        """
        X_train = pd.DataFrame({
            'user': self.train_df["userId"].map(self.user_to_index),
            'movie': self.train_df["movieId"].map(self.movies_to_index)
        })
        y_train = self.train_df["rating"].astype(np.float32)
        return np.asarray(X_train), np.asarray(y_train)

    def generate_testset(self):
        """
        테스트 데이터셋을 생성

        Returns:
            tuple: 사용자와 영화 인덱스로 구성된 배열, 평점 배열
        """
        X_test = pd.DataFrame({
            'user': self.test_df["userId"].map(self.user_to_index),
            'movie': self.test_df["movieId"].map(self.movies_to_index)
        })
        y_test = self.test_df['rating'].astype(np.float32)
        return np.asarray(X_test), np.asarray(y_test)

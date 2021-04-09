# core_model
elmo+cnn+bilstm model

-------------

### 실행
1. docker pull (change_y_to_numpy.py실행할 필요 없음)
2. pad_vector_*.npy 파일 생성
  ```
  python print_np.py
  ```
3. 모델 생성
  ```
  python core-model/core_model.py
  ```
4. 모델 성능 측정
  ```
  python core-model/predict_core_model.py
  ```
-------------

#### 오픈소스

##### 1. elmo_embedding
https://github.com/ratsgo/embedding

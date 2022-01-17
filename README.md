# recommendation
추천 시스템

<이론 정리>
: Factorization Machines
Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International conference on data mining (pp. 995-1000). IEEE.
https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db
https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

크게 두 가지 접근법이 자주 사용됨
1) CBF : Content-Based Filtering
  = user나 item의 features의 벡터들을 사용
2) CF : Collaborative Filtering
  = 과거 user의 행동들로만 작동 -> domain-free의 장점을 가짐

- Matrix Factorization
 - CF 접근법들은 대부분 MF(Matrix Factorization)에 근간을 둠
   (2006-2009 Netflix Prize 때문에 유명해짐)
 [핵심 기제]  
 - sparse user-item interaction matrix를 user와 item embedding을 포함한 두 개의 low-rank matrix의 approximate product으로 factorized하는 것
 - 이렇게 만들어진 latent factors를 학습하고 나면, 유사도를 구할 수 있고, 관측되지 않은 선호도도 추론해낼 수 있음
 - 관찰된 것과 예측된 rating의 오차제곱합을 최소화하도록 알고리즘을 학습시킴
 [단점]
 - 그런데 MF는 raw user behavior를 다룰 때 잘 작동하지 못할 수 있음
 - latent user 선호도간의 높은 관련성이 있을 때 잘 작동하지 못함
 - 보조적인 features를 포함시키지 못함 -> cold-star problem
 - binary rating을 사용해야하므로 user 선호도를 간접적으로 볼 수 밖에 없음

- Factorization Machines
  - arbitrary real-valued features를 저차원의 latent factor space로 mapping하는 generic supervised 학습 모델
  - 회귀, 분류, ranking의 예측 문제 등에 광범위하게 적용될 수 있음
  - real-valued features과 numeric target variables의 tuples을 user-item interaction으로 표상 가능
  - equation을 살펴보면 2차 선형 회귀와 비슷한 모습, but factorized interaction 모수를 사용함
    = feature interaction weights는 두 feature의 latent factor space embeddings의 내적으로 나타남
    -> 추정해야할 모수를 대폭 줄여줌
    

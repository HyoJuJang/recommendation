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
    
https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
https://www.jefkine.com/recsys/2017/03/27/factorization-machines/
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

[Optimaization]
stochastic gradient descent(SGD)
	- Gradient descent : Batch Gradient Descent, SGD, Mini Batch GD
		- SGD가 많이 쓰이는데, step size와 같은 모수들 직접 tune해야 함
		이 tuning 자동으로 하게 하는 것 중 가장 성공적인 스키마 = AdaGrad
		좀 더 informative gradient-based learning임
			- AdaGrad도 두 가지 버전 : Diagonal / Full AdaGrad
	- Adam : classical SGD 대신 쓸 수 있는 optimization 알고리즘(https://arxiv.org/abs/1412.6980)
		- 직관적이고 효율적이며 noisy/sparse gradients 문제에도 쓰일 수 있음
		- 하이퍼파라미터가 직관적으로 해석될 수 있으며 튜닝이 거의 필요 없음
		- AdaGrad + RMSProp의 장점을 섞어놓음
		- default parameters recommended(Jason Brownlee on July 3, 2017)
		TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
		Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
		Blocks: learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
		Lasagne: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
		Caffe: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
		MxNet: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
		Torch: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
		alternating least squares(ALS)
		Markov Chain Monte Carlo(MCMC)

[Loss function]
for classification : logit or hinge loss
	- binary cross-entropy loss, hinge loss, squared hinge loss, multi-class , ...
for regression : least square loss with L2 regularization
	- mean squared, mean squared logarithmic, mean absolute error
	
[Parameters]
latent factors 수(k) :  With a higher k, you have more specific categories.
learning rate(SGD)

---
layout: post
title: Self Attention에 대해 공부
description: >
  Transformer에 기초, 핵심 부분에 대해 알아보자
image: /assets/img/post/self_attention/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Self Attention에 대해 공부

트랜스포머 메커니즘이 등장하면서 자연어 처리와 기계 번역의 전통적인 패러다임은 근본적으로 변화하게 되었다. 특히, 이러한 혁신을 주도하는 핵심 기술 중 하나가 셀프 어텐션(**Self Attention**)이다. **Self Attention**은 텍스트 데이터 내의 단어 간 상호 작용과 관계를 정교하게 모델링할 수 있어, 문장의 의미를 훨씬 정확하게 파악하거나 번역하는 데 있어 비약적인 향상을 가져왔다. 이러한 **Self Attention** mechanism이 어떻게 동작하는지, 그리고 왜 이것이 효과적인지에 대한 깊은 이해는 자연어 처리를 연구하거나 애플리케이션을 개발하는 데 있어 필수적이다. 본 글에서는 이러한 점들을 깊이 있고 자세하게 탐구하며, **Self Attention**의 개념부터 그 구현 디테일, 그리고 그 장점까지 철저히 분석하고자 한다.

## **1. 셀프 어텐션의 개념**

Self-attention, 즉 자기 주의 메커니즘은 입력 시퀀스의 각 요소(보통은 단어의 임베딩)가 다른 요소와 어떻게 상호 작용하는지를 모델링한다. 이는 Query, Key, Value라는 세 가지 다른 표현으로 분해된다. Query는 현재 단어가 어떤 정보에 주의를 기울여야 하는지를 나타내고, Key는 각 단어가 어떤 정보를 가지고 있는지를 나타내며, Value는 실제로 그 정보를 어떻게 활용할지 나타낸다.

## **2. 셀프 어텐션의 동작 원리**

**Self Attention**은 세 가지 주요 단계로 이루어진다. 각 단계에서는 입력 문장 내의 단어들 간의 상대적인 중요도를 계산하여 가중치를 부여한다.

### **2.1. 어텐션 스코어 계산**

첫 번째 단계에서는 쿼리(Query), 키(Key), 밸류(Value)라는 세 가지 vector를 생성한다. 이때, query는 현재 단어에 대한 정보를 담고 있고, key와 value는 입력 문장의 다른 단어들에 대한 정보를 담고 있다. query와 key vector 간의 유사도를 측정하여 attention score를 계산한다.

우선 Query와 Key의 내적(dot product)을 계산하여 attention score를 얻는다. 이 score는 Query와 Key 사이의 유사도를 나타내며, 높을수록 더 많은 주의를 기울여야 함을 의미한다.

$$
Attention\;Score = Query\;\cdot\;Key^{T}
$$

### **2.2. 소프트맥스(softmax) 함수를 통한 가중치 계산**

두 번째 단계에서는 attention score를 확률 분포로 변환하기 위해 소프트맥스 함수를 적용한다. 이렇게 하면 각 단어의 중요도에 대한 가중치를 얻을 수 있다. 이 가중치는 Query와 Key사이의 상대적인 중요도를 나타낸다. 즉 가중치가 높을수록 해당 단어는 문장의 의미를 이해하는 데에 더 큰 영향을 미친다.

$$
Weight\;=\;Softmax(Attention\;Score)
$$

### **2.3. 가중치를 적용한 밸류 계산**

세 번째 단계에서는 소프트맥스 함수를 통해 얻은 가중치를 Value vector에 곱샘 적용하여 최종적인 단어 표현을 계산한다. 즉 각 단어의 value vector를 가중치와 곱한 뒤 합산하여 **Self Attention**의 결과를 얻게 된다. 이를 통해 문맥을 고려한 표현을 얻을 수 있다.

$$
Output\;=\;Weights\times value
$$

### 2.4. 계산 과정 예제

전체적인 어텐션 메커니즘은 주어진 Query, Key, Value에 대해 다음과 같은 연산을 수행한다:

$$
Attention(Q,K,V)\;=\;softmax(\frac{QK^{T}}{\sqrt{d_{k}}})\,V
$$

여기서 d_k 는 Key의 차원이다. 분모에 d_k의 제곱근을 사용하는 이유는 계산값이 너무 커짐을 방지하기 위함이다.

예를 통해 계산 과정을 한번 훑어 보겠다.

$$
Q = \begin{bmatrix} 1\quad 0 \\0\quad 1 \end{bmatrix} \\
K = \begin{bmatrix}1\quad 2\\3\quad 4 \end{bmatrix} \\
V = \begin{bmatrix} 0\quad 1 \\1\quad 0 \end{bmatrix} \\


$$

다음과 같이 초기 Q,K,V가 정의되었다고 가정하고 진행하겠다.

먼저 Query 와 key를 내적한다. 여기서 key의 차원인 d_k 는 2다. (2차원 matrix)

$$
QK^{T}\;=\; \begin{bmatrix}1\quad0\\0\quad1\end{bmatrix}
\begin{bmatrix}1\quad2\\3\quad4\end{bmatrix}^{2}
\;=\;
\begin{bmatrix}1\quad2\\2\quad3\end{bmatrix}
$$

그 다음 내적한 값에 d_k의 제곱근으로 나눈 후,

$$
\frac{QK^{T}}{\sqrt{2}}
\;=\;
\begin{bmatrix}0.70710678\quad1.41421356\\1.41421356\quad2.12132034\end{bmatrix}
$$

softmax를 적용하여 가중치를 계산한다.

$$
Weight\;=\;softmax(\frac{QK^{T}}{\sqrt{2}})
\;=\;
\begin{bmatrix}0.33023845\quad0.66976155\\0.33023845\quad0.66976155\end{bmatrix}
$$

마지막으로 계산된 Weight를 Value값에 계산하면 첫 번째 value vector계산이 끝이난다.

$$
Weight\;\times\;Value
\;=\;
\begin{bmatrix}0.66976155\quad0.33023845\\0.66976155\quad0.33023845\end{bmatrix}
$$

위 예제 계산은 매우매우 간단히 설명하기 위해 2x2 matrix를 이용했지만, 실제론 더 높은 차원의 matrix와 더 많은 self-attention layer를 사용하기 때문에 훨씬 복잡한 계산을 하게 된다.

Transformer의 핵심 포인트인 Self-attention은 위 와같은 Self-Attention 계산을 여러 layer를 통해 수행할 수 있는데, 그이유는 아래와 같이 정리할 수 있다.

1. 관점의 다양성

- 각 셀프 어텐션 레이어는 입력에 대해 다양한 "관점"에서 정보를 추출할 수 있다. 예를 들어, 첫 번째 레이어는 단어 간의 국소적인 관계를 파악하고, 두 번째 레이어는 좀 더 넓은 문맥에서의 관계를 파악하는 형식이다. 여러 개의 레이어를 쌓음으로써 모델은 입력 데이터의 복잡한 특성과 관계를 더 정확하게 학습할 수 있다.

2. 계층적 학습의 특성

- 딥 러닝 모델, 특히 심층 신경망은 데이터의 계층적 특성을 학습하는 능력이 있다. 즉, 하위 레이어는 간단한 특성을, 상위 레이어는 더 복잡한 특성을 학습한다. 셀프 어텐션 레이어를 여러 개 쌓음으로써, 모델은 이러한 계층적 특성을 더 효과적으로 학습할 수 있다.

그리고 결과적으로 loss function에 의해 최적의 value를 찾기위해 반복 계산을 한다.

위 계산을 수행하는 python code는 아래와 같이 매우 간단하게 구현할 수 있다.

```python
import numpy as np
from scipy.special import softmax

# Define Query, Key, and Value matrices
Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 2], [2, 3]])
V = np.array([[0, 1], [1, 0]])

# Calculate the dot product of Q and K.T (transpose of K)
QK_T = np.dot(Q, K.T)

# Divide by the square root of d_k (dimension of Key vectors)
d_k = K.shape[1]
QK_T_scaled = QK_T / np.sqrt(d_k)

# Apply the Softmax function
weights = softmax(QK_T_scaled, axis=-1)

# Multiply the weights with Value matrix
output = np.dot(weights, V)

print(" QK_T : ", QK_T)
print("\n QK_T_scaled : ", QK_T_scaled)
print("\n weights : ", weights)
print("\n output : ", output)
```

## **3. Self Attention의 장점**

**Self Attention**은 다른 기법들과 비교하여 다음과 같은 장점을 가지고 있다.

### **3.1. 병렬 계산 가능**

전통적인 RNN이나 LSTM은 시퀀스의 각 요소를 순차적으로 처리해야 하지만, **Self Attention**은 모든 요소를 동시에 처리할 수 있다. 이로 인해 병렬 계산이 가능하며, 연산 속도가 빠르다.

### **3.2. 문장 길이에 영향 받지 않음**

**Self Attention**은 문장의 길이에 영향을 받지 않는다. 기존의 순차적인 모델들은 문장의 길이가 길어질수록 계산 시간이 길어지는 단점이 있었다. 하지만 **Self Attention**은 모든 단어 간의 관계를 한 번에 계산하기 때문에 문장의 길이에 상관 없이 일정한 계산 시간을 유지할 수 있다.

### **3.3. 문맥 파악 능력 강화**

**Self Attention**은 문장 내의 각 단어가 다른 단어와 어떻게 상호 작용하는지를 정밀하게 파악한다. 이로 인해 문맥을 더 잘 이해할 수 있으며, 이는 특히 고정된 윈도우 크기의 CNN이나 단기 메모리의 RNN보다 뛰어나다.

## **4. 마무리**

**Self Attention**은 트랜스포머 아키텍처의 핵심 요소로, 복잡한 문장 구조와 의미를 효과적으로 모델링할 수 있다. 이 메커니즘을 정확히 이해하고 활용하면, 자연어 처리와 관련된 다양한 문제에 대해 더욱 효과적인 솔루션을 개발할 수 있다.

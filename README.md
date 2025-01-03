# Attention Is All You Need

> RNN과 CNN 계열의 모델에서 병렬처리의 어려움, Sequence가 긴 데이터에 대한 장거리 의존성 부족 및 기하급수적으로 증가하는 연산량을 해결하기 위해서 등장하였습니다.
> 해당 논문에서는 최초로 Self Attention 구조만을 사용하여 기계번역을 구현하였습니다. RNN 구조를 갖고 있지 않기 때문에 거리가 먼 x에 대해서도 모두 동일한 연산량으로 계산할 수 있어 계산의 효율성을 높히면서도 높은 성능을 보여주었습니다.

#### [ Input Data ]
Encoder에서는 우선 데이터를 Embedding하게 됩니다.

예를들어 입력 데이터가 "I am happy " 라는 입력이 들어오면 Encoder에서 해당 데이터를 우선 **Token 단위** 로 데이터를 분할하게 됩니다. 예를 들어서 단어 단위로 Token을 분할하게 되면  Token = ['I', 'am', 'happy'] 이 됩니다.

이후 각 Token들에 대해서 특정 차원 D_model 으로 임베딩 하게 됩니다.
예를들어 D_model = 3 인경우 
Token = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 이런식으로 Embedding이 되게 됩니다.
해당 논문에서는 D_model = 512로 임베딩하였다고 합니다.

그렇다면 Token의 수가 N 개이고 D_model = 512 로 입력 데이터를 임베딩 하는경우 N * 512 크기의 Matrix를 얻게 됩니다. 이것이 Input 데이터가 됩니다.

조금만 더 자세히 알아보도록 하겠습니다.
사실 Transformer에서는 단어 단위로 토큰을 생성하지 않았습니다. BPE 라는 알고리즘을 활용해서 알파벳 단위로 토큰을 설정하게 됩니다.
예를들어 "I am Happy" 라는 문장이 들어오게 되면 ['i', 'a', 'm', 'h', 'a', 'p', 'p', 'y' ] 로 분할하게 됩니다. 그리고 나서 반복적으로 각 문장에 동시에 등장하는 알파벳 끼리 묶어주게 됩니다.
![BPE Tokenization](images/image.png)
<br><br>
위의 사진과 같이 각 알파벳 별 동시에 등장하는 횟수를 파악하고 가장 많이 등장하는 쌍을 하나로 묶어주게 됩니다. 이를 통해서 원하는 K번 만큼 반복하게 되고 이떄 묶여진 것이 토큰이 되게 됩니다.
즉, Transformer에서는 토큰을 생성하는 방식을 BPE 방식을 사용해서 토큰을 생성하였습니다.
그리고 Sentence가 N개를 동시에 처리할때 문장 마다 포함하고 있는 토큰의 수가 다르게 됩니다. 이때 행렬로 input을 넣어야 하기 때문에 길이를 맞춰야하는데 이때 길이를 맞출수 있는 방법은 간단하게 2가지가 존재할 수 있습니다.
1. 특정 길이를 설정해서 해당 길이로 긴것은 자르고 짧은 것은 padding을 한다.
2. 가장 많은 토큰 수를 기준으로 부족한 것을 padding으로 채우는 방법이 존재할 수 있습니다.
<br>
이렇게 해서 최종적으로 Transformer에서는 ( 배치 사이즈 X 토큰의 갯수 X 모델의 차원 ) 의 행렬이 들어가게 됩니다.

하지만 Sequence 데이터에서 " 내가 너를 때린다 " VS "너가 나를 떄린다 " 처럼 각 단어의 순서가 중요한 경우가 존재합니다. 그렇기에 각 데이터의 순서에 대한 정보를 입력해줄 필요가 있습니다.
그래서 해당 논문에서는 Positional Encoding값을 더해줌으로써 각 Token의 위치 정보를 추가해주게 됩니다. 이때 Positional Encoding 값 또한 ( 배치사이즈 X N X 512 ) 크기가 됩니다. 

positional encoding의 경우에는 상대적 절대적 위치를 판단할수 있는 특정 값을 더하게 됩니다. 이때 단순히 행렬을 더하기 함으로써 구현 되는데 이를 통해서 자연스럽게 Matrix에 위치 정보가 들어가게 됩니다. 하지만 너무나도 큰 위치 정보 값을 더하게 된다면 실제 임베딩 값의 의미가 무효화 되기 때문이 -1 과 1 사이의 값을 갖는 sin과 cos를 활용해서 구현되어있습니다. 이렇게 만들어진 행렬에서 Q, K, V를 만들어 이를 Attention 하게 된다면 자연스럽게 위치 정보가 들어가 있게 됩니다. 그래서 의미와 위치 정보를 통한 Attention이 이뤄지게 됩니다.
<br>
![Positional Encoding](images/image-1.png)

#### [ Attention ]
위에서 만든 ( 배치 X N X 512 )  크기의 행렬을 Input이 Encoder의 Attention Module에 들어가게 됩니다.

Attention 구조에서는 데이터를 Q, K, V 행렬로 분할하여 계산하게 됩니다.
![Multi-Head Attention](images/image-2.png)

위의 수식과 같이 Attention이 계산되는데, 이때 필요한 것이 바로 Q, K, V 값입니다. 이 값을 만들기 위해서 W_iq, W_ik, W_iv 라는 가중치 행렬을 두게 됩니다.

(**해당 설명에서는 Multi Head의 개념으로 설명하겠습니다, Multihead를 사용하는 이유는 서로다른 가중치를 사용함으로써 보다 다양한 Insight를 얻을수 있기 때문입니다.**)

Multi Head = 8인경우 보통 dim = d_model / head 로 결정되기 때문에 예시에서는 dim = 512/ 8 = 64 차원으로 가중치 행렬의 차원을 결정하겠습니다.

즉, Q를 만들기 위한 W_iq의 경우 W_1q, W_2q ..... W8_q 까지 8개의 가중치가 필요하며 최종적으로 Q,K,V를 위해서는 24개의 행렬의 필요하게 됩니다. 각 행렬의 크기는 [512 * 64]가 됩니다.

아무튼, 각 행렬을 Encoder의 Input과 행렬곱을 하게 됩니다. 그러면 [ N * 512 ] @ [ 512 * 64 ] 를 통해서 최종적으로 [ N * 64 ] 크기의 Q, K, V가 각각 8개씩 만들어 지게 됩니다.

최종적으로 정리해보면 다음과 같이 각 8개씩 [ N * 64 ] 크기의 Q, K, V가 생성되게 됩니다. 그리고 이해를 돕기 위해서 가장 앞에 있는 Q, K, V 만을 사용해서 설명해보도록 하겠습니다. ( 어차피 모두 동일하게 작동하기에 마지막에 concat하면 되기에 )

<br>

![Attention Mechanism](images/image-3.png)

<br>
이제 Q, K, V에 대해 구했으니 위의 공식에 맞게 계산을 하게 됩니다. 가장 먼저 Q @ K.T를 진행하게 됩니다. 이를 통해서 [ N * 64] @ [ 64 * N ] = [ N * N ]의 행렬을 얻게 되는데, 해당 행렬이 의미하는것은 각 토큰들 끼리의 유사도(연관성) 을 나타내게 됩니다. 위에서 들었던 예시를 생각해보면 3 * 3에서 각 I 와 am, happy의 유사도를 구하게 된다고 생각하면 됩니다.

그리고 dim이 크게 되면 자연스럽게 값이 너무 커지기 때문에 sqrt(D_model) 값으로 나눠주어 너무 큰 값이 되지 않도록 정규화를 진행해 줍니다.

그리고 softmax()를 진행하여 상대적인 유사도를 파악하게 됩니다. 이렇게 되면 서로 연관성이 높은 토큰들의 값이 높고, 그렇지 않으면 작은 값을 갖게 됩니다. 

이후 각 토큰의 정보인 V에 곱해 줌으로써, 각 토큰 정보에 **가중치**를 부여하여 더 중요한 토큰의 정보가 반영되고 덜 중요한 단어는 영향을 덜 받게 됩니다.

이러한 계산을 통해서 서로 연관성이 높은 토큰들에 대해서는 높은 가중치를 갖게되며, 관련성이 없는 토큰들에 대해서는 낮은 가중치를 갖게됩니다.
이를 행렬의 관점에서 바라보게 되면 다음과 같습니다.

    softmax([N * 64] @ [ 64 * N]) @ [N * 64] => [ N * 64 ] 

즉, Attention input과 output은 동일한 크기를 갖게 됩니다. 우리가 MultiHead를 사용하였기에 각 Q,K,V가 64차원으로 8개씩 추가적으로 있기에 이를 모두 concat 시킨다면 결국 N * 512 크기의 행렬을 얻게 됩니다. 

<br>

![Multi-Head Attention Output](images/image-4.png)
<br>
즉, Attention의 input과 output의 크기가 동일하기 떄문에 해당 논문에서는 Attention 구조를 총 6개 쌓아올리는 구조를 만들수 있었습니다.

그리고 아래와 같은 구조를 총 6번을 진행하게 됩니다. FFN의 경우 행렬에 비선형성을 추가해주는 역할을 하고 Norm & ADD애서 정규화 및 skip connection을 하게 됩니다.
<br>

![Stacked Layers](images/image-5.png) 

< FFN 수식 : 2개의 가중치와 ReLU를 거치게 된다.>

[ Multi_Head Attention -> Norm & ADD -> FFN -> Norm & ADD ] 

<br>

> **Decoder**
<br>

![Decoder Structure](images/image-8.png)

Decoder는 학습시에는 Ground Truth를 입력으로 받게 됩니다.
예를들어 우리가 "나는 학교에 간다" 라는 한국어를 영어로 번역하는 모델을 만드는 경우 "I", "I am" , "I am going" .. 이런 순서대로 input을 받게 됩니다. 이를 통해서 순차적으로 각 단어에 대해 학습하게 됩니다.
이렇게 input으로 GT를 넣는 방법을 **Teacher Forcing** 이라고 한다. Teacher Forcing은 효율적이고 빠르지만, inference에서는 GT를 사용할 수 없고 모델이 추론한 토큰을 기반으로 값을 예측해야하기에 Train, Inference에서 오는 차이가 발생할 수 있다고 합니다.

아무튼 Decoder에서는 GT를 정답으로 넣게 되는데, 이때 shifted right를 적용하여 Input을 넣는다고 합니다.

여기서 의미하는 **shifted right** 란 말 그대로 시퀀스 데이터를 오른쪽으로 이동시킨 개념입니다.

> Decoder의 입력 : [sos] I am going to school 

> Decoder의 목표 : I am going to school [eos]

즉, 입력값을 오른쪽으로 한 칸씩 밀어서 모델에서 [sos]가 I를 예측하고 I가 am을 예측하는 식으로 각 단어들이 다음의 단어를 예측할 수 있도록 모델을 학습시키기 위해서 다음과 같은 방식으로 Input 데이터를 변형하게 됩니다.

이를 통해서 디코더가 현재 단어를 예측하는 경우 이전의 단어에 대한 정보만을 활용하여 다음 단어를 예측할 수 있도록 할 수 있기 때문입니다.
<br>

![Masked Multi-Head Attention](images/image-6.png)

Shifted right : 입력 단계에서 이전 단어만 사용 가능하게 하여 모델 자체가 이전 단어를 통해 다음 단어를 예측하도록 만들어줍니다.

Masked Multi-head attention : Attention 진행 시 미래의 데이터를 볼 수 없게 하는 것입니다.

즉, right shift에서 이전 단어만을 사용해서 데이터를 예측하게 하였고, Attention step에서도 자기 이전의 데이터들만 가지고 attention을 진행하도록 미래의 단어(다음 단어) 들에 대해 Masking 처리를 진행하게 됩니다.

이는 Matrix 형태를 Lower Triangular Matrix로 만들어 미래 단어들에 대해서 모두 -inf로 처리하게 됩니다.

아래 표에서 'x' 표시가 바로 Attention 가능을 의미합니다. 그래서 I 의 경우 [sos], I 와 Attention이 가능하고, school의 경우 모든 데이터와 Attention이 가능하게 됩니다.
<br>

![Attention Masking](images/image-6.png)

**Shifted right** : 입력 단계에서 이전 단어만 사용 가능하게 하여 모델 자체가 이전 단어를 통해 다음 단어를 예측하도록 만들어줍니다.

**Masked Multi-head attention** : Attention 시 미래의 데이터를 통해 단어를 예측하는 경우 이는 정보 손실로 판단합니다. 즉, Inference 단계에서 미래의 데이터가 없기에 Train 과정에서도 정보 유출을 방지하기 위하여 사용되었습니다.

Input data -> Masked Multihead Attention -> Add & Norm을 거치게 됩니다.

이후 Decoder에서는 또 한번의 Attention을 거치게 됩니다.

이때 Decoder-Encoder Attention을 사용합니다. 

이때 Input으로 Q, K, V를 입력하게 되는데, K, V 의 경우 Encoder의 최종 output의 K, V 를 input으로 받게 됩니다. 그리고 Q의 경우 Decoder의 이전 Layer의 output을 받아오게 됩니다.

Query는 디코더의 현재 예측에 필요한 정보를 담고 있습니다. 이렇게 함으로써 Query가 디코더의 현재 상태를 반영하고, Key와 Value가 소스 시퀀스의 정보를 반영하도록 분리함으로써, 디코더는 현재 예측에 필요한 소스 시퀀스의 관련 정보를 효과적으로 검색할 수 있게 됩니다.


# ViT 추가 내용
ViT는 Transformer의 입력으로 이미지를 사용한 모델입니다. 해당 논문에서는 단순히 입력값을 패치로 나누어 embedding하고 CLS 토큰을 추가하는것 이외에는 Transformer의 Encdoer와 동일한 구조를 갖습니다.
그렇다면 왜 ViT가 잘 작동할까요?????

지금까지 CNN이 잘 작동했던 이유는 1. Spatial Locality 2.Positional Invaraince 입니다. 즉, 주변 픽셀을 보고 판단하라고 사전에 조건을 걸어두어 더 적은 연산량을 갖게 되고, 객체가 이미지내 어디에 존재하든 인식할수 있어서 FCN 보다 빠르게 수렴하고 높은 정확도를 낼수 있었습니다. 하지만 ViT의 경우 이러한 가정이 없이 단순히 data와 연산량을 통해서 model이 스스로 어디가 중요한지를 학습하게 되면서 CNN에서는 인식할 수 없었던 예외적인 상황또한 고려할 수 있게 되어 보다 높은 성능을 낼수 있게 되었습니다. 즉 Inductive Bias (귀납적 편향 : 인간이 정해둔 가정? ) 를 최대한 줄이면서 모델이 더 많은 정보를 학습할수 있도록 유도하였습니다.

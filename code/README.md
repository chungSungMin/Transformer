# Encoder

> 입력값 설명

Batch Size : 32<br>
Sequence : 8<br>
Embedding Dimention : 512<br>
Num_head : 8<br>

즉 입력 데이터의 형태는 ( 32, 8, 512) 를 갖게 됩니다.



> StackedEncoder class 설명

nn.ModuleListe()함수를 통해서 Encoder 클래스를 우리가 원하는 만큼 반복하도록 한 리스트 자체를 self.StackedEncoder로 설정하게 됩니다.<br>
이후 forward()에서 반복문을 통해서 우리가 원하는 만큼 입력값이 Encoder를 통과하도록 합니다.





> MultiHead Attention Class 설명

< **init 함수 설명** > <br>
각 매계변수들을 self.을 활용하여 선언합니다.<br>
그리고 각 가중치 행렬들을 서언해줍니다.<br><br>

self.d_model : multihead 사용시 각 head의 차원 <br>
-> 만일 모델의 차원이 num_head로 나누어 떨어지지 않는다면 에러가 발생합니다. <br>

self.WQ : Q를 만드는 선형변환  <br>
self.WK : K를 만드는 선형변환 <br>
self.WV : V를 만드는 선형변환 <br>
self.fc_out : attention value를 구한 후 수행되는 FFN 층 <br>
<br>


< **MultiHead Attention Forward 설명** >

입력값의 Batch_size, Sequence, Dimention을 unpacking 합니다. <br>

이후 입력값을 활용해서 Q, K, V 를 생성합니다. <br>

**이후 MultiHead Attention을 하기 위해 Q,K,V를 변형시킵니다.D 차원의 모델 차원을 (num_head , d_model) 로 나누어 차원을 추가하게됩니다.** <br>

이렇게 view()를 통해서 만들어진 데이터의 크기는 ( Batch size, Sequence, num_head, d_model) 이 됩니다. 그리고 Attention 연산을 수행 할 때 필요한 값인 Sequence와 d_model의 값을 사용하기 위해서 .transpose(1,2) 값을 사용해서 ( Batch size, num_head, Sequence, d_model ) 이 되게 됩니다.

최종적으로 모든 Q, K, V는 multihead를 수행하기 위해서 **( Batch size, num_head, Sequence, d_model )** 크기를 갖게 됩니다.

이후 Attention Score를 계산하기 위해서 torch.matmul(Q, K.transpose(-1,-2)) 계산을 진행합니다. transpose(-1, -2)를 진행한 이유는 Seqence와 d_model의 행렬곱 연산을 진행해야 되기 때문입니다. matmul()의 경우 자동으로 index가 가장 큰 값 2개에 대해서 행렬 곱을 수행하게 됩니다.

이렇게 계산이 되면 데이터의 크기는 ( Batch Size, Num_head, Sequence, Sequence )가 됩니다.
그리고 이를 softmax를 취하고 d_model의 차원으로 정규화를 진행하기 위해서 F.softmax()와 np.sqrt() 로 정규화를 진행합니다. softmax()를 취하는 이유는 모든 가중치들을 확률 값으로 근사해야 하기에 0~1 사이의 값을 갖도록 해주기 위해서 사용됩니다.

이렇게 얻은 Attention Score와 V를 matmul()을 진행하게 되면 최종적으로 Attention Value를 얻을수 있게 됩니다.

이렇게 얻어진 Attention Value의 경우 ( Batch size, Num_head, Sequence, d_model )의 크기를 갖게 되고 Encoder의 경우 Input과 output의 크기가 동일하게 나와야 하기에 이를 복원시키기 위해서 다시 Num_head와 sequence의 자리를 변경한 후에 이를 ( Batch_size, Sequence, Dimention )의 크기로 변경해주게 됩니다.

그리고 입력값인 Residual과 단순 summation을 통해 skip connection을 하고, LayerNorm()을 한후 결과를 Return 해줍니다.

<br>

> FeedForward Class 설명

in_channels와 ffn_expand를 입력으로 받아 512차원을 2048로 확장하여 다양한 패턴을 학습 할수 있게 한후 ReLU()함수를 통해 비선형성을 추가합니다. 
그런 다음 다시 Linear()를 통해서 2048차원을 512차원으로 만들어줍니다. 이렇게 FFN층을 나온다음 Residual과 단순 더하기를 한 다음 LayerNorm()을 적용한후 결과를 반환합니다.



<br><br>

# Decoder

Batch Size : 32<br>
Sequence : 8<br>
Embedding Dimention : 512<br>
Num_head : 8<br>



즉 입력 데이터의 형태는 ( 32, 8, 512) 를 갖게 됩니다.
**Decoder의 경우 입력값을 2개로 필요한 경우가 존재합니다.** <br> 바로 Encoder-Decoder MultiHead Attention을 구현하기 위한 Encoder의 K,V와 Decoder의 입력 자체를 게산한 Q 가 필요하게 됩니다.
> stackedDecoder class 설명 <br>

stackedDecoder의 경우 Encoder와 동일하게 Decoder를 우리가 원하는 num_layer만큼 반복 하도록 해줍니다.
> Masked MultiHead Attention class 설명 <br>

Masked MultiHead Attention의 경우 현재 상태보다 이전의 상태에 대해서 -inf를 사용해서 0값으로 만들어줘야합니다. 그렇게 하기 위해서 torch.matmul(Q, K.transpose(-1,-2))의 결과인 S * S 크기를 갖는 Lower Traiangular Matrix를 만들어주게 됩니다. 값은 모두 1로 채워주게 됩니다. 그리고 masked_fill() 함수를 통해서 우리가 구한 torch.matmul(Q, K.transpose(-1,-2))를 정삼각 행렬로 만들어 주게 됩니다. 이렇게 함으로써 우리는 현재 index보다 작은 값만 살리기 때문에 Auto Regressive 특징을 살릴 수 있게 됩니다. <br> 이후 과정은 일반적인 MutliHead Attention 과정과 동일하게 진행이 됩니다.
> Encoder Deocer MultiHead Attention class 설명 <br>

우선 기본적으로 일반 MultiHead Attention과 init 함술들 동일하게 만들어줍니다. 우리가 필요로 하는 Q,K,V 가중치 행렬들을 설정하게 됩니다.<br>
그리고 Forward 부분이 중요합니다.
**Forward에서는 Encoder의 결과와, 이전 레이어에서 넘어온 Masked MultiHead Attention의 결과를 받아옵니다.** 이를 통해서 Q의 값은 Deocder의 결과를, K,V는 Encoder의 결과를 사용하면서 순차적인 정보만 사용하여 Auto Regressive 특징을 살리면서 문맥의 전체적인 정보가 담겨있는 K,V를 사용하여 2가지 특징을 모두 고려하여 Attention을 계산할 수 있게 됩니다.


> FeedForward class 설명 <br>

feedforward 클래스의 경우 encoder와 동일하기 때문에 넘어가도록 하겠습니다.

<br><br>

# Transformer

> Transformer class 설명 <br>

위에서 구현한 Encoder와 Decoder를 통해서 Transformer를 동작시킵니다.
init()에서 우리가 만든 encoder, decoder를 객체로 생성합니다.
그리고 forward()에서 encoder를 통해서 source값의 특징을 추출하게 되고, decoder에는 target 값과 encoder의 결과를 넣어주게 됩니다.
그리고 Transformer를 num_layers = 3로 설정하고 실행 시킨 결과는 아래와 같습니다.

<br>

<div align="center">
    <img src="image-1.png" alt="alt text">
</div>

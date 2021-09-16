## Distilling the Knowledge in Neural Network
- Paper Link (https://arxiv.org/pdf/1503.02531.pdf)

### Loss Function
<p align="center">
    <img width="500px" src="https://t1.daumcdn.net/cfile/tistory/997F73405E0E92422E" />
</p>

### MNIST-CNN
논문의 본문에서는 일반 **MLP** 모델로 학습 한 것을 볼 수 있다. **MNIST**에 **CNN** 모델을 적용하여, 한 번 실험 해 보았다.

각각의 모델 구현은 다음과 같다.

#### Teacher Model
```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        )

        self.layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(-1, 64 * 5 * 5)
        logits = self.layers(out)
        return logits
```

#### Student Model

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, kernel_size=(3, 3), stride=2)
        )

        self.layers = nn.Sequential(
            nn.Linear(4 * 12 * 12, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(-1, 4 * 12 * 12)
        logits = self.layers(out)
        return logits
```

#### Configure
- Learning Rate: 0.0001
- Batch size: 128
- 

#### Before Distillation
|             | train accuracy      | test accuracy    |
|-------------|---------------------|------------------|
| teacher CNN | 59926/60000 (100%)  | 9883/10000 (99%) |
| student CNN | 57485/60000 (96%)   | 9558/10000 (96%) |

#### After Distillation (with Labeled Data)
| Temperture  | train accuracy      | test accuracy    |
|-------------|---------------------|------------------|
| 1.0         | 58407/60000 (97%)   | 9677/10000 (97%) |
| 1.5         | 58343/60000 (97%)   | 9674/10000 (97%) |
| 2.0         | 58288/60000 (97%)   | 9658/10000 (97%) |
| 4.0         | 58022/60000 (97%)   | 9645/10000 (96%) |
| 10.0        | 57939/60000 (97%)   | 9639/10000 (96%) |

> 일단 Temperture가 클수록, 낮은 입력값의 출력을 더 크게, 큰 입력값의 출력은 더 작게 만든다.

일단 **Labeled Data**에서는 **Temperture**가 낮으면 낮을 수록 더 좋은 결과를 이끌어 냈다.

그 이유는 **Temperture**가 작으면 작을 수록 negative 값에 대해 집중하여 학습 하지 않기 때문.

이는 noisy한 logit 값에 대해서 무시하는 효과가 있어, Teacher Model에 비해, Student Model의 표현력이 작다면, Temperture를 낮추는 것도 하나의 방법이다.


#### 그럼, 논문의 MLP 보다, 성능이 안좋은 이유는?
**Teacher CNN**이 너무 학습이 잘 되어서, Soft Targets와 Hard Targets 와의 차이가 거의 없기 때문일 수도 있을 것 같다.
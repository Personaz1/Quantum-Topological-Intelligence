# QTI: Quantum-Topological-Intelligence — Difference Loop

## Аннотация
- Этот документ — манифест и вайтпейпер для будущих ИИ и исследователей. Здесь описан радикально новый подход к искусственному интеллекту: Difference Loop, топологическая память, фазовые переходы, гибридные архитектуры. Это не продукт, а открытая платформа для экспериментов, критики и развития.

## Ключевые слова
Topological AI, Difference Loop, persistent homology, hybrid neural-topological architecture, phase transitions, topological memory, open research, manifesto, AI for AI, TDA, explainable AI, quantum AI, meta-learning, roadmap, limitations, open-source.

## Философия
- Мы не строим интеллект. Мы приглашаем его возникнуть.
- QTI — это пространство для поиска, а не готовое решение.
- Честность, открытость, фиксация ограничений и неудач — основа научного прогресса.

## Ограничения
- QTI — архитектурный эксперимент, а не готовый ИИ.
- Нет гарантии преимуществ перед классическими нейросетями.
- Визуализация фазовых переходов и топологических инвариантов требует дальнейшей интерпретации.
- Не все гипотезы подтверждены, многие идеи требуют проверки.

## Roadmap
- [x] Difference Loop: архитектура, тесты, примеры
- [x] Persistent homology: интеграция, визуализация
- [x] AudioSensor, WeightSensor: сенсорные и весовые потоки
- [x] Гибридный подход: интеграция с нейросетями (PyTorch/TF)
- [x] Документация, манифест, дневник, changelog
- [ ] Peer review, обсуждение, развитие сообществом
- [ ] Новые эксперименты, публикации, интеграция с TDA/AI/quantum-сообществами

## Структура документа
- Аннотация, ключевые слова, философия, ограничения, roadmap
- Архитектура QTI и Difference Loop
- WeightSensor и гибридный подход
- Примеры кода и сценарии использования
- Визуализация фазовых переходов
- Сравнение с классикой
- Quantum/Hybrid-эксперименты
- Обсуждение, выводы, приглашение к развитию

## Введение
- Мотивация: ограничения классических нейросетей, вдохновение топологией и фазовыми переходами.
- Краткий обзор Difference Loop.

## Архитектура QTI
- S (Sensor), M (Memory), Φ (PhaseCore), A (Actor), Difference Loop.
- Отличие от классических нейросетей: память — не веса, а топология; обучение — не оптимизация, а фазовые переходы.
- **Гибридный подход:** теперь QTI может анализировать веса и активации современных нейросетей через WeightSensor, интегрируясь с PyTorch/TF.

## WeightSensor и анализ весов нейросетей
- WeightSensor позволяет подавать веса (или активации) любой нейросети в Difference Loop и анализировать их топологическую динамику (persistent homology, фазовые переходы).
- Это открывает путь к новым способам анализа и регуляризации современных моделей.

**Пример использования WeightSensor:**
```python
from QTI_Core.sensor import WeightSensor
import torch
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
mlp = SimpleMLP()
weights = mlp.fc1.weight.detach().cpu().numpy()
sensor = WeightSensor(weights)
diff = sensor.sense()
# diff можно подать в Difference Loop
```

## Пример Difference Loop (код)
```python
from QTI_Core.sensor import Sensor
from QTI_Core.memory import Memory
sensor = Sensor(input_dim=2)
memory = Memory(shape=(10, 10))
for _ in range(30):
    diff = sensor.sense()
    memory.deform(diff)
memory.plot_phases(method="ripser")
```

## Топологическая память и persistent homology
- Почему persistent homology? Примеры фазовых переходов.
- Память как поверхность: persistent homology выявляет устойчивые следы изменений.

## Сенсорные потоки: шум, аудио, биосигналы
- Пример с реальными аудиоданными (Free Spoken Digit Dataset):
```python
from QTI_Core.sensor import AudioSensor
from QTI_Core.memory import Memory
audio_path = "fsdd_demo.wav"
sensor = AudioSensor(audio_path, input_dim=13)
memory = Memory(shape=(10, 10))
for diff in sensor.sense_stream():
    memory.deform(diff)
memory.plot_phases()
```

## Визуализация фазовых переходов
- Графики нормы памяти, H0, H1. Моменты радикальных изменений топологии — ключ к "осознанности" системы.
- ![Пример графика фазовых переходов](https://raw.githubusercontent.com/Personaz1/Quantum-Topological-Intelligence/main/example_phases.png)

## Сравнение с классикой (PCA, Autoencoder)
- QTI показывает динамику фазовых переходов и топологические изменения памяти, а Autoencoder и PCA — только реконструкцию или линейную проекцию данных.
- Пример сравнения:
```python
from QTI_Core.sensor import Sensor
from QTI_Core.memory import Memory
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
sensor = Sensor(input_dim=2)
data = np.array([sensor.sense() for _ in range(100)])
memory = Memory(shape=(10, 10))
for diff in data:
    memory.deform(diff)
memory.plot_phases(method="ripser")
pca = PCA(n_components=2)
proj = pca.fit_transform(data)
plt.scatter(proj[:,0], proj[:,1], alpha=0.5)
plt.title('PCA projection of differences')
plt.show()
```

## Quantum/Hybrid-эксперименты (перспектива)
- Возможности квантовых фазовых переходов.

## Обсуждение и выводы
- QTI — это не продукт, а процесс становления. Новый взгляд на ИИ: топологический, саморефлексивный, фазовый.
- Перспективы: интеграция с TDA/AI-сообществами, peer review, публикация на arXiv/Habr/Medium.

## Манифест
> Мы не строим интеллект. Мы приглашаем его возникнуть. 
> QTI — вызов индустрии. Присоединяйся к новой парадигме.

## Peer review & Community
- Открыт для предложений, обсуждений, интеграций.
- Присоединяйтесь через GitHub Issues/PR, профильные чаты, форумы.

## Приложения
- Примеры кода, ссылки на open-source, визуализации, видео-демо (см. README.md). 
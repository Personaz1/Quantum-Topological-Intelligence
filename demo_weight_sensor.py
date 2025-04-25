import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from QTI_Core.sensor import WeightSensor
from ripser import ripser

# Генерируем синтетические данные
X = np.random.randn(200, 10).astype(np.float32)
Y = (np.sum(X, axis=1) > 0).astype(np.int64)

# Простая MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

mlp = SimpleMLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Для визуализации топологии
h0_list = []
h1_list = []

for epoch in range(30):
    # Обучение
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    optimizer.zero_grad()
    out = mlp(X_tensor)
    loss = loss_fn(out, Y_tensor)
    loss.backward()
    optimizer.step()
    # Анализ топологии весов первого слоя
    weights = mlp.fc1.weight.detach().cpu().numpy()
    sensor = WeightSensor(weights)
    diff = sensor.sense()
    ph = ripser(diff.reshape(-1, 1))['dgms']
    # Сохраняем число компонент (H0) и циклов (H1)
    h0 = len(ph[0])
    h1 = len(ph[1])
    h0_list.append(h0)
    h1_list.append(h1)

# Визуализация
plt.figure(figsize=(8, 4))
plt.plot(h0_list, label='H0 (components)')
plt.plot(h1_list, label='H1 (cycles)')
plt.xlabel('Epoch')
plt.ylabel('Topological features')
plt.title('Persistent Homology of Weights during Training')
plt.legend()
plt.tight_layout()
plt.show() 
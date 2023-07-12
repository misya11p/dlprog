# dlprog

*Deep Learning Progress*

A Python library for progress bars with the function of aggregating each iteration's value.  
It helps manage the loss of each epoch in deep learning or machine learning training.

## Installation

```bash
pip install dlprog
```

## Basic Usage

Setup

```python
from dlprog import Progress
prog = Progress()
```

Example

```python
import random
import time
n_epochs = 5
n_iter = 10

prog.start(n_epochs=n_epochs, n_iter=n_iter) # Initialize start time and epoch.
for _ in range(n_epochs):
    for _ in range(n_iter):
        time.sleep(0.1)
        value = random.random()
        prog.update(value) # Update progress bar and aggregate value.
```

Output

```
1/5: ######################################## 100% [00:00:01.05] value: 0.47299
2/5: ######################################## 100% [00:00:01.05] value: 0.56506
3/5: ######################################## 100% [00:00:01.05] value: 0.51024
4/5: ######################################## 100% [00:00:01.03] value: 0.58471
5/5: ######################################## 100% [00:00:01.04] value: 0.60167
```

## In deep learning training

Setup

```python
from dlprog import train_progress
prog = train_progress()
```

Example. Case of training a deep learning model with PyTorch.

```python
n_epochs = 5
n_iter = len(dataloader)

prog.start(n_epochs=n_epochs, n_iter=n_iter)
for _ in range(n_epochs):
    for x, label in dataloader:
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        prog.update(loss.item())
```

Output

```
1/5: ######################################## 100% [00:00:03.08] loss: 0.34099
2/5: ######################################## 100% [00:00:03.12] loss: 0.15259
3/5: ######################################## 100% [00:00:03.14] loss: 0.10684
4/5: ######################################## 100% [00:00:03.15] loss: 0.08008
5/5: ######################################## 100% [00:00:03.20] loss: 0.06347
```

If you want to obtain exact values considering batch size:

```python
prog.update(loss.item(), weight=len(x))
```

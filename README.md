# dlprog

*Deep Learning Progress*

[![PyPI](https://img.shields.io/pypi/v/dlprog)](https://pypi.org/project/dlprog/1.0.0/)

<br>

A Python library for progress bars with the function of aggregating each iteration's value.  
It helps manage the loss of each epoch in deep learning or machine learning training.

## Installation

```bash
pip install dlprog
```

## General Usage

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

## Optional Arguments

### `leave_freq`

The frequency of leaving the progress bar.

```python
n_epochs = 12
n_iter = 10
prog.start(n_epochs=n_epochs, n_iter=n_iter, leave_freq=4)
for _ in range(n_epochs):
    for _ in range(n_iter):
        time.sleep(0.1)
        value = random.random()
        prog.update(value)
```

Output

```
 4/12: ######################################## 100% [00:00:01.06] loss: 0.34203 
 8/12: ######################################## 100% [00:00:01.05] loss: 0.47886 
12/12: ######################################## 100% [00:00:01.05] loss: 0.40241 
```

### `unit`

Multiple epochs as a unit.

```python
n_epochs = 12
n_iter = 10
prog.start(n_epochs=n_epochs, n_iter=n_iter, unit=4)
for _ in range(n_epochs):
    for _ in range(n_iter):
        time.sleep(0.1)
        value = random.random()
        prog.update(value)
```

Output

```
  1~4/12: ######################################## 100% [00:00:04.21] value: 0.42546 
  5~8/12: ######################################## 100% [00:00:04.23] value: 0.48436 
 9~12/12: ######################################## 100% [00:00:04.22] value: 0.54864 
```

## Version History

### [1.0.0](https://pypi.org/project/dlprog/1.0.0/) (2023-07-13, Latest)
- Add `Progress` class.
- Add `train_progress` function.

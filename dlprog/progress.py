import time
from typing import Optional, Union


SPM = 60 # seconds per minute
MPH = 60 # minutes per hour
def time_format(t: float) -> str:
    h = t // (MPH * SPM)
    m = t // MPH
    s = t % MPH
    return f'{int(h):02}:{int(m):02}:{s:05.2f}'


Number = Union[int, float]

class TrainProgress:
    def __init__(
        self,
        n_iter: int = None,
        n_epochs: int = None,
        width: int = 50,
        symbol: str = '#',
    ):
        self.n_iter = n_iter
        self.n_epochs = n_epochs
        self.width = width
        self.symbol = symbol
        self.reset()

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self.is_training = False
        self.now_epoch = 1
        self._epoch_reset()

    def _epoch_reset(self):
        self.now_iter = 0
        self.prop = 0.
        self.epoch_loss = 0
        self._loss_den = 0
        self.start_time = time.time()
        self.now_time = self.start_time
        self._make_epoch_text()

    def _make_epoch_text(self):
        epoch_text = str(self.now_epoch)
        if self.n_epochs:
            epoch_text = epoch_text.rjust(len(str(self.n_epochs)))
            epoch_text += f'/{self.n_epochs}'
        self._epoch_text = epoch_text

    def start(self, **kwargs):
        self.set(**kwargs)
        assert self.n_iter is not None, '"n_iter" is not set.'
        self.is_training = True
        self.now_epoch = 1
        self._epoch_reset()

    def _draw_bar(self):
        index_text = f'{self._epoch_text}:'
        bar_text = (self.symbol * int(self.width * self.prop))
        bar_text = bar_text.ljust(self.width)
        prop_text = f'{int(self.prop * 100)}%'.rjust(4)
        time_text = f'[{time_format(self.now_time - self.start_time)}]'
        loss_text = f'loss: {self.epoch_loss / self._loss_den:.4f}'
        text = ' '.join([
            index_text,
            bar_text,
            prop_text,
            time_text,
            loss_text
        ])
        print(text, end='\r', flush=True)

    def update(
        self,
        loss: Optional[float] = None,
        weight: Number = 1,
        advance: int = 1,
        auto_step: bool = True,
    ):
        if not self.is_training:
            self._start()

        self.now_iter += advance
        if loss is not None:
            self.epoch_loss += weight * loss
            self._loss_den += weight
        self.prop = min(self.now_iter / self.n_iter, 1)
        self.now_time = time.time()
        self._draw_bar()
        if self.now_iter >= self.n_iter and auto_step:
            self.step()

    def step(self):
        print()
        self.now_epoch += 1
        self._epoch_reset()

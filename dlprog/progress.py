import time
from typing import Optional, Union


SPM = 60 # seconds per minute
MPH = 60 # minutes per hour
def time_format(t: float) -> str:
    h = t // (MPH * SPM)
    m = t // MPH
    s = t % MPH
    return f'{int(h):>02}:{int(m):>02}:{s:05.2f}'


Number = Union[int, float]

class TrainProgress:
    def __init__(
        self,
        n_iter: int,
        width: int = 50,
        symbol: str = '#'
    ):
        self.n_iter = n_iter
        self.width = width
        self.symbol = symbol

    def _draw_bar(self):
        bar_text = (self.symbol * int(self.width * self.prop))
        bar_text = bar_text.ljust(self.width)
        prop_text = f'{int(self.prop * 100)}%'.rjust(4)
        time_text = time_format(self.now_time - self.start_time)
        loss_text = f'loss: {self.epoch_loss / self._loss_den:.4f}'
        text = f'{self._epoch_text}: {bar_text} {prop_text} | {time_text} | {loss_text}'
        print(text, end='\r', flush=True)

    def _make_epoch_text(self):
        epoch_text = str(self.now_epoch)
        if self.n_epochs:
            epoch_text = epoch_text.rjust(len(str(self.n_epochs)))
            epoch_text += f'/{self.n_epochs}'
        self._epoch_text = epoch_text

    def _epoch_start(self):
        self.now_iter = 0
        self.prop = 0.
        self.epoch_loss = 0
        self.now_epoch += 1
        self._loss_den = 0
        self.start_time = time.time()
        self.now_time = self.start_time
        self._make_epoch_text()

    def start(self, n_epochs: Optional[int] = None):
        self.now_epoch = 0
        self.n_epochs = n_epochs
        self._epoch_start()

    def update(self, loss: Optional[float] = None, weight: Number = 1):
        self.now_iter += 1
        if loss is not None:
            self.epoch_loss += weight * loss
            self._loss_den += weight
        self.prop = min(self.now_iter / self.n_iter, 1)
        self.now_time = time.time()
        self._draw_bar()

    def step(self):
        print()
        self._epoch_start()
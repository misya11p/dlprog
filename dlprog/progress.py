from typing import Optional, Union

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

    def _draw_bar(self, prop: float):
        bar_text = (self.symbol * int(self.width * prop)).ljust(self.width)
        prop_text = f'{int(prop * 100)}%'.rjust(5)
        loss_text = f'loss: {self.epoch_loss / self._loss_den:.4f}'
        text = f'{self._epoch_text}: {bar_text} {prop_text} | {loss_text}'
        print(text, end='\r', flush=True)

    def _make_epoch_text(self):
        epoch_text = str(self.now_epoch)
        if self.n_epochs:
            epoch_text = epoch_text.rjust(len(str(self.n_epochs)))
            epoch_text += f'/{self.n_epochs}'
        self._epoch_text = epoch_text

    def _epoch_start(self):
        self.now_iter = 0
        self.epoch_loss = 0
        self.now_epoch += 1
        self._loss_den = 0
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
        prop = min(self.now_iter / self.n_iter, 1)
        self._draw_bar(prop)

    def step(self):
        print()
        self._epoch_start()
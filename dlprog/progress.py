from typing import Optional


class TrainProgress:
    def __init__(
        self,
        n_iter: int,
        width: int = 30,
        symbol: str = '█'
    ):
        self.n_iter = n_iter
        self.width = width
        self.symbol = symbol

    def _draw_bar(self, prop: float):
        n_symbols = int(self.width * prop)
        bar_text = f'{self._epoch_text}: {self.symbol * n_symbols}'
        print('\r', bar_text, end='', flush=True)

    def _make_epoch_text(self):
        epoch_text = str(self.now_epoch)
        if self.n_epochs:
            epoch_text += f'/{self.n_epochs}'
        self._epoch_text = epoch_text

    def _epoch_start(self):
        self.now_iter = 0
        self.epoch_loss = 0
        self.now_epoch += 1
        self._make_epoch_text()

    def start(self, n_epochs: Optional[int] = None):
        self.now_epoch = 0
        self.n_epochs = n_epochs
        self._epoch_start()
        self._draw_bar(0)

    def update(self, loss: float):
        self.now_iter += 1
        self.epoch_loss += loss
        prop = min(self.now_iter / self.n_iter, 1)
        self._draw_bar(prop)

    def step(self, del_bar: bool = True):
        if del_bar:
            print('\r', '  ' * self.width, end='\r', flush=True)
        else:
            print()
        loss_text = f'loss: {self.epoch_loss / self.now_iter:.4f}'
        print(self._epoch_text, loss_text, sep=', ', flush=True)
        self._epoch_start()
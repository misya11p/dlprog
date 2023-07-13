from .utils import time_format
import time
from typing import Optional, Union, Callable


Number = Union[int, float]

class Progress:
    def __init__(
        self,
        n_iter: Optional[int] = None,
        n_epochs: Optional[int] = None,
        agg_fn: Union[None, str, Callable[[Number, Number], Number]] = 'mean',
        label: Optional[str] = 'value',
        width: int = 40,
        symbol: str = '#',
        leave_freq: int = 1,
    ):
        """
        Progress bar class.
        When the following attributes are None, the progress bar is not 
        displayed correctly.

        Args:
            n_iter (int):
                Number of iterations per epoch. Defaults to None.
            n_epochs (int):
                Number of epochs. Defaults to None.
            agg_fn (Union[str, Callable[[Number, Number], Number]]):
                Aggregation function for epoch value with weight.
                Defaults to 'mean'.
            label (str):
                Label for progress bar. Defaults to 'value'.
            width (int):
                Width of progress bar. Defaults to 40.
            symbol (str):
                Symbol for progress bar. Defaults to '#'.
            leave_freq (int):
                Frequency of leaving the progress bar. If <= 0, none are
                left. Defaults to 1.
        """
        self.n_iter = n_iter
        self.n_epochs = n_epochs
        self.agg_fn = agg_fn
        self.label = label
        self.width = width
        self.symbol = symbol
        self.leave_freq = leave_freq
        self.reset()

    _agg_fns = {
        'mean': lambda s, w: s / w,
        'sum': lambda s, w: s,
    }

    def _set_agg_fn(self):
        if isinstance(self.agg_fn, str):
            self._agg_fn = self._agg_fns[self.agg_fn]
        else:
            self._agg_fn = self.agg_fn

    def set(self, **kwargs):
        """Set attributes."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._set_agg_fn()

    def reset(self):
        """Reset attributes. """
        self.is_training = False
        self.now_epoch = 1
        self._text_length = 0
        self._epoch_reset()
        self._set_agg_fn()

    def _epoch_reset(self):
        self.now_iter = 0
        self.prop = 0.
        self.epoch_value = 0
        self._epoch_value_weight = 0
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
        """Start training. Initialize start time and epoch."""
        self.set(**kwargs)
        assert self.n_iter is not None, '"n_iter" is not set.'
        self.is_training = True
        self.now_epoch = 1
        self._epoch_reset()

    def _draw(self):
        index_text = f'{self._epoch_text}:'
        bar_text = self.symbol * int(self.width * self.prop)
        bar_text = bar_text.ljust(self.width)
        prop_text = f'{int(self.prop * 100)}%'.rjust(4)
        time_text = f'[{time_format(self.now_time - self.start_time)}]'
        value_text = f'{self.label}: ' if self.label else ''
        if self._epoch_value_weight:
            value = self._agg_fn(self.epoch_value, self._epoch_value_weight)
        else:
            value = 0.
        value_text += f'{value:.5f}'
        text = ' '.join([
            index_text,
            bar_text,
            prop_text,
            time_text,
            value_text
        ])
        print('\r' + ' ' * self._text_length, end='')
        print('\r' + text, end='', flush=True)
        self._text_length = len(text)

    def update(
        self,
        value: Optional[float] = None,
        weight: Number = 1,
        advance: int = 1,
        auto_step: bool = True,
    ):
        """
        Update progress bar and aggregate value.

        Args:
            value (Optional[float]):
                value. If None, only the progress bar advances. Defaults
                to None.
            weight (Number, optional):
                weight of value. Defaults to 1.
            advance (int, optional):
                Number of iterations to advance. Defaults to 1.
            auto_step (bool, optional):
                If True, step() is called when the number of iterations
                reaches n_iter. Defaults to True.
        """
        if not self.is_training:
            self._start()

        self.now_iter += advance
        if value is not None:
            self.epoch_value += weight * value
            self._epoch_value_weight += weight
        self.prop = min(self.now_iter / self.n_iter, 1)
        self.now_time = time.time()
        self._draw()
        if self.now_iter >= self.n_iter and auto_step:
            self.step(not self.now_epoch % self.leave_freq)

    def step(self, leave: bool = True):
        """
        Step to the next epoch.
        
        Args:
            leave (bool):
                If True, leave the progress bar. Defaults to True.
        """
        if leave:
            print('\n', end='')
        else:
            print('\r', ' ' * self._text_length, end='\r')
        self.now_epoch += 1
        self._epoch_reset()


def train_progress(
    n_iter: int = None,
    n_epochs: int = None,
    width: int = 40,
    symbol: str = '#',
) -> Progress:
    """
    Return a progress bar for deep learning training.

    Returns:
        Progress: Progress bar object.
    """
    return Progress(
        n_iter=n_iter,
        n_epochs=n_epochs,
        label='loss',
        width=width,
        symbol=symbol,
    )

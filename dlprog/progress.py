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
        unit: int = 1,
    ):
        """
        Progress bar class.
        When the following attributes are None, the progress bar is not 
        displayed correctly.
        Attributes defined in this constructor will be default values.

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
            unit (int):
                Unit of progress bar (epoch). Defaults to 1.
        """
        self._defaults = {
            'n_iter': n_iter,
            'n_epochs': n_epochs,
            'agg_fn': agg_fn,
            'label': label,
            'width': width,
            'symbol': symbol,
            'leave_freq': leave_freq,
            'unit': unit,
        }
        self.reset()

    _agg_fns = {
        'mean': lambda s, w: s / w,
        'sum': lambda s, w: s,
    }

    def _reset_attr(self):
        """Set attributes to default values."""
        for k, v in self._defaults.items():
            setattr(self, k, v)
        self._set_attr()

    def set_defaults(self, **kwargs):
        """Modify default values."""
        for k, v in kwargs.items():
            assert k in self._defaults, f'"{k}" is an invalid attribute.'
            self._defaults[k] = v

    def _set_agg_fn(self):
        if isinstance(self.agg_fn, str):
            self._agg_fn = self._agg_fns[self.agg_fn]
        else:
            self._agg_fn = self.agg_fn

    def _set_unit(self):
        self._unit = max(1, int(self.unit))

    def _set_n_epochs(self):
        self._n_digits = len(str(self.n_epochs))

    def _set_attr(self):
        """Set internal attributes."""
        self._set_agg_fn()
        self._set_unit()
        self._set_n_epochs()

    def set(self, **kwargs):
        """Define attributes."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._set_attr()

    def reset(self):
        """Reset all attributes."""
        self.is_running = False
        self.now_epoch = 0
        self.n_bar = 0
        self._text_length = 0
        self.values = []
        self._reset_attr()
        self._set_attr()
        self._epoch_reset()
        self._bar_reset()

    def _epoch_reset(self):
        """Reset attributes for epoch."""
        self.now_iter = 0
        self.prop = 0.
        self.value = 0
        self.value_weight = 0
        self.start_time = time.time()
        self.now_time = self.start_time
        self._make_epoch_text()

    def _bar_reset(self):
        """Reset attributes for progress bar."""
        self._bar_now_iter = 0
        self._bar_prop = 0.
        self._bar_value = 0
        self._bar_value_weight = 0
        self._bar_start_time = time.time()
        self._bar_now_time = self.start_time

    def _make_epoch_text(self):
        """Make epoch text."""
        if self.n_epochs:
            if self._unit >= 2:
                first = (self.n_bar - 1) * self._unit + 1
                last = self.n_bar * self._unit
                epoch_text = f'{first}-{last}'
                epoch_text = epoch_text.rjust(self._n_digits * 2 + 1)
            else:
                epoch_text = str(self.now_epoch).rjust(self._n_digits)
            epoch_text += f'/{self.n_epochs}'
        else:
            epoch_text = str(self.now_epoch)
        self._epoch_text = epoch_text

    def start(self, **kwargs):
        """
        Start running. Initialize start time and epoch. You can set the
        attributes to be used at this runtime. If not set, the default
        value is used.
        """
        self.reset()
        self.set(**kwargs)
        assert self.n_iter is not None, '"n_iter" is not set.'
        self.is_running = True
        self.now_epoch = 1
        self.n_bar = 1
        self._make_epoch_text()

    def _draw(self):
        """Draw progress bar."""
        index_text = f'{self._epoch_text}:'
        bar_text = self.symbol * int(self.width * self._bar_prop)
        bar_text = bar_text.ljust(self.width)
        prop_text = f'{int(self._bar_prop * 100)}%'.rjust(4)
        bar_time = self._bar_now_time - self._bar_start_time
        time_text = f'[{time_format(bar_time)}]'
        value_text = f'{self.label}: ' if self.label else ''
        if self._bar_value_weight:
            value = self._agg_fn(self._bar_value, self._bar_value_weight)
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
        print('\r' + text, end=' ', flush=True)
        self._text_length = len(text)

    def _update_values(self, advance, value, weight):
        """Update values."""
        self.now_iter += advance
        self._bar_now_iter += advance
        if value is not None:
            self.value += weight * value
            self.value_weight += weight
            self._bar_value += weight * value
            self._bar_value_weight += weight
        self.prop = self.now_iter / (self.n_iter * self._unit)
        self._bar_prop = self._bar_now_iter / (self.n_iter * self._unit)
        self.now_time = time.time()
        self._bar_now_time = time.time()

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
        assert self.is_running, 'Progress bar is not started. Call start().'
        self._update_values(advance, value, weight)
        self.prop = self.now_iter / self.n_iter
        self._draw()
        if auto_step and self.prop >= 1.:
            bar_step = self._bar_prop >= 1.
            leave = not (self.n_bar % self.leave_freq)
            leave = self.leave_freq > 0 and leave
            self.step(bar_step=bar_step, leave=leave)

    def step(self, bar_step: bool = True, leave: bool = True):
        """
        Step to the next epoch.
        
        Args:
            leave (bool):
                If True, leave the progress bar. Defaults to True.
        """
        if bar_step:
            if leave:
                print('\n', end='')
            else:
                print('\r', ' ' * self._text_length, end='\r')
            self.n_bar += 1
            self._bar_reset()
        self.values.append(self._agg_fn(self.value, self.value_weight))
        self.now_epoch += 1
        self._epoch_reset()


def train_progress(**kwargs) -> Progress:
    """
    Return a progress bar for machine learning training.

    Returns:
        Progress: Progress bar object.
    """
    kwargs['label'] = 'loss'
    return Progress(**kwargs)

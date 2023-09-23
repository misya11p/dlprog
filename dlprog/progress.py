from .utils import time_format
import time
from typing import Optional, Union, Callable, List


Number = Union[int, float]


class Progress:
    def __init__(
        self,
        n_iter: Optional[int] = None,
        n_epochs: Optional[int] = None,
        label: Optional[Union[str, List[str]]] = None,
        n_values: int = 1,
        agg_fn: Union[None, str, Callable[[Number, Number], Number]] = 'mean',
        width: int = 40,
        symbol: str = '#',
        leave_freq: int = 1,
        unit: int = 1,
        note: str = '',
        sep: str = ', '
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
            label (Optional[Union[str, List[str]]]):
                Label for progress bar. If you want to use multiple
                values, input labels as an iterable. Defaults to None.
            n_values (int):
                Number of values to be aggregated. If this number
                differs from the number of labels, the number of labels
                is used. Use this when you want to set None to label.
                Defaults to 1.
            agg_fn (Union[str, Callable[[Number, Number], Number]]):
                Aggregation function for epoch value with weight.
                Defaults to 'mean'.
            width (int):
                Width of progress bar. Defaults to 40.
            symbol (str):
                Symbol for progress bar. Defaults to '#'.
            leave_freq (int):
                Frequency of leaving the progress bar. If <= 0, none are
                left. Defaults to 1.
            unit (int):
                Unit of progress bar (epoch). Defaults to 1.
            note (str):
                Note for progress bar. Defaults to ''.
            sep (str):
                Separator character for values and note. Defaults to 
                ', '.
        """
        self._defaults = {
            'n_iter': n_iter,
            'n_epochs': n_epochs,
            'label': label,
            'n_values': n_values,
            'agg_fn': agg_fn,
            'width': width,
            'symbol': symbol,
            'leave_freq': leave_freq,
            'unit': unit,
            'note': note,
            'sep': sep,
        }
        self.reset()

    _agg_fns = {
        'mean': lambda s, w: s / w,
        'sum': lambda s, w: s,
    }

    def _set(self, **kwargs):
        """Define attributes."""
        args = self._defaults.copy()
        args.update(kwargs)
        for k, v in args.items():
            setattr(self, k, v)
        self._set_attr()

    def _set_attr(self):
        """Set internal attributes."""
        # Set aggregation function
        if isinstance(self.agg_fn, str):
            self._agg_fn = self._agg_fns[self.agg_fn]
        else:
            self._agg_fn = self.agg_fn

        # Set unit
        self._unit = max(1, int(self.unit))

        # Set n_digits
        self._n_digits = len(str(self.n_epochs))

        # Set labels
        if self.label is None:
            self._labels = ['' for _ in range(self.n_values)]
        elif isinstance(self.label, str):
            self._labels = [self.label]
        else:
            self._labels = self.label
        self.n_values = len(self._labels)

    def set_defaults(self, **kwargs):
        """Modify default values."""
        for k, v in kwargs.items():
            assert k in self._defaults, f"'{k}' is an invalid attribute."
            self._defaults[k] = v
        self._set()

    def reset(self, **kwargs):
        """Reset all attributes."""
        self.is_running = False
        self.now_epoch = 0
        self.n_bar = 0
        self._text_length = 0
        self.values = []
        self._set(**kwargs)
        self._epoch_reset()
        self._bar_reset()

    def _epoch_reset(self):
        """Reset attributes for epoch."""
        self.now_iter = 0
        self.prop = 0.
        self._epoch_values = [0 for _ in range(self.n_values)]
        self._epoch_value_weights = [0 for _ in range(self.n_values)]
        self.start_time = time.time()
        self.now_time = self.start_time
        self._epoch_note = self.note
        self._make_epoch_text()

    def _bar_reset(self):
        """Reset attributes for progress bar."""
        self._bar_now_iter = 0
        self._bar_prop = 0.
        self._bar_values = [0 for _ in range(self.n_values)]
        self._bar_value_weights = [0 for _ in range(self.n_values)]
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
        self.reset(**kwargs)
        assert self.n_iter is not None, '"n_iter" is not set.'
        self.is_running = True
        self.now_epoch = 1
        self.n_bar = 1
        self._keep_step = False
        self._epoch_note = self.note
        self._make_epoch_text()

    def _draw(self):
        """Draw progress bar."""
        index_text = f'{self._epoch_text}:'
        bar_text = self.symbol * int(self.width * self._bar_prop)
        bar_text = bar_text.ljust(self.width)
        prop_text = f'{int(self._bar_prop * 100)}%'.rjust(4)
        bar_time = self._bar_now_time - self._bar_start_time
        time_text = f'[{time_format(bar_time)}]'
        value_texts = []
        for label, value, weight in zip(
            self._labels, self._bar_values, self._bar_value_weights
        ):
            value_text = f'{label}: ' if self.label else ''
            if weight:  # Avoid ZeroDivisionError
                value = self._agg_fn(value, weight)
            else:
                value = 0.
            value_text += f'{value:.5f}'
            value_texts.append(value_text)
        value_text = self.sep.join(value_texts)
        text = ' '.join([
            index_text,
            bar_text,
            prop_text,
            time_text,
            value_text,
        ])
        if self._epoch_note:
            text += self.sep + self._epoch_note
        print('\r' + ' ' * self._text_length, end='')
        print('\r' + text, end=' ', flush=True)
        self._text_length = len(text)

    def _update_values(self, advance, value, weight):
        """Update values."""
        self.now_iter += advance
        self._bar_now_iter += advance
        if value is not None:
            if isinstance(value, (int, float)):
                value = [value]
            if isinstance(weight, (int, float)):
                weight = [weight for _ in range(self.n_values)]
            for i in range(self.n_values):
                self._epoch_values[i] += weight[i] * value[i]
                self._epoch_value_weights[i] += weight[i]
                self._bar_values[i] += weight[i] * value[i]
                self._bar_value_weights[i] += weight[i]

        self.prop = self.now_iter / (self.n_iter * self._unit)
        self._bar_prop = self._bar_now_iter / (self.n_iter * self._unit)
        self.now_time = time.time()
        self._bar_now_time = time.time()

    def update(
        self,
        value: Optional[Union[Number, List[Number]]] = None,
        weight: Union[Number, List[Number]] = 1,
        advance: int = 1,
        auto_step: bool = True,
        note: Optional[str] = None,
        defer: bool = False,
    ):
        """
        Update progress bar and aggregate value.

        Args:
            value (Optional[Union[Number, List[Number]]], optional):
                value. If None, only the progress bar advances. Defaults
                to None.
            weight (Union[Number, List[Number]], optional):
                weight of value. Defaults to 1.
            advance (int, optional):
                Number of iterations to advance. Defaults to 1.
            auto_step (bool, optional):
                If True, step() is called when the number of iterations
                reaches n_iter. Defaults to True.
            note (Optional[str], optional):
                Note for progress bar. Defaults to None.
            defer (bool, optional):
                If True, auto-step will be deferred until the next
                memo() call. Use when you want to update a note at the
                end of the epoch. Defaults to False.
        """
        assert self.is_running, 'Progress bar is not started. Call start().'
        self._update_values(advance, value, weight)
        self.prop = self.now_iter / self.n_iter
        if note is not None:
            self._epoch_note = note
        self._draw()
        if auto_step and self.prop >= 1.:
            bar_step = self._bar_prop >= 1.
            leave = not (self.n_bar % self.leave_freq)
            leave = self.leave_freq > 0 and leave
            if defer:
                self._keep_step = True
                self._keep_step_info = {
                    'bar_step': bar_step,
                    'leave': leave,
                }
            else:
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
        value = [self._agg_fn(v, w) for v, w in zip(
            self._epoch_values, self._epoch_value_weights)]
        if self.n_epochs == 1:
            value = value[0]
        self.values.append(value)
        self.now_epoch += 1
        self._epoch_reset()

    def memo(self, note: str):
        """
        Change note text for progress bar.

        Args:
            note (str): Text.
        """
        self._epoch_note = note
        self._draw()
        if self._keep_step:
            self._keep_step = False
            self.step(**self._keep_step_info)


def train_progress(**kwargs) -> Progress:
    """
    Return a progress bar for machine learning training. The label just
    defaults to 'loss'.

    Returns:
        Progress: Progress bar object.
    """
    if 'label' not in kwargs:
        kwargs['label'] = 'loss'
    return Progress(**kwargs)

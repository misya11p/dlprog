from .utils import time_format, value_format
import time
from typing import Optional, Union, Callable, List
import warnings


Number = Union[int, float]
Numbers = Union[Number, List[Number]]


class Progress:
    def __init__(
        self,
        n_iter: Optional[int] = None,
        n_epochs: Optional[int] = None,
        label: Optional[Union[str, List[str]]] = None,
        n_values: int = 1,
        agg_fn: Union[None, str, Callable[[Number, Number], Number]] = "mean",
        momentum: Optional[float] = None,
        width: int = 40,
        leave_freq: int = 1,
        unit: int = 1,
        defer: bool = False,
        note: str = "",
        symbol: str = "#",
        round: int = 5,
        sep_label: str = ": ",
        sep_values: str = ", ",
        sep_note: str = ", ",
        store_all_values: bool = True,
        store_all_times: bool = True,
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
                is used. Use this when you want to set None to label or
                when you want to use multiple values with the same
                label. Defaults to 1.
            agg_fn (Union[str, Callable[[Number, Number], Number]]):
                Aggregation function for epoch value with weight.
                Defaults to 'mean'.
            momentum (float):
                Momentum for updating values. If None, the value is
                updated by the aggregation function. If set, the value
                is updated by the exponential moving average, agg_fn and
                weight are ignored. Defaults to None.
            width (int):
                Width of progress bar. Defaults to 40.
            leave_freq (int):
                Frequency of leaving the progress bar. If <= 0, none are
                left. Defaults to 1.
            unit (int):
                Unit of progress bar (epoch). Defaults to 1.
            defer (bool):
                If True, auto-step will be deferred until the next
                memo() call. Use when you want to update a note at the
                end of the epoch. Defaults to False.
            note (str):
                Note for progress bar. Defaults to ''.
            symbol (str):
                Symbol for progress bar. Defaults to '#'.
            round (int):
                Number of digits to round to. Default is 5.
            sep_label (str):
                Separator character for value and label.
                Defaults to ': '.
            sep_values (str):
                Separator character for values. Defaults to ', '.
            sep_note (str):
                Separator character for note. Defaults to ', '.
            store_all_values (bool):
                If True, store values of all iterations. Set to False if
                you want to reduce memory usage. Defaults to True.
            store_all_times (bool):
                If True, store times of all iterations. Set to False if
                you want to reduce memory usage. Defaults to True.
        """
        self._defaults = {
            "n_iter": n_iter,
            "n_epochs": n_epochs,
            "label": label,
            "n_values": n_values,
            "agg_fn": agg_fn,
            "momentum": momentum,
            "width": width,
            "leave_freq": leave_freq,
            "unit": unit,
            "defer": defer,
            "note": note,
            "symbol": symbol,
            "round": round,
            "sep_label": sep_label,
            "sep_values": sep_values,
            "sep_note": sep_note,
            "store_all_values": store_all_values,
            "store_all_times": store_all_times,
        }
        self.reset()

    _agg_fns = {
        "mean": lambda s, w: s / w,
        "sum": lambda s, w: s,
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

        # Check unit
        assert isinstance(self.unit, int), "'unit' must be an integer."
        assert self.unit > 0, "'unit' must be greater than 0."

        # Set epoch num of digits
        self._n_epoch_digits = len(str(self.n_epochs))

        # Set labels
        if self.label is None:
            self._labels = ["" for _ in range(self.n_values)]
        elif isinstance(self.label, str):
            self._labels = [self.label for _ in range(self.n_values)]
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
        self._all_values = []
        self._all_times = []
        self._set(**kwargs)
        self._epoch_reset()
        self._bar_reset()

    def _epoch_reset(self):
        """Reset attributes for epoch."""
        self.now_iter = 0
        self.prop = 0.
        self._epoch_values = [0 for _ in range(self.n_values)]
        self._epoch_value_weights = [0 for _ in range(self.n_values)]
        self._now_epoch_values = [0 for _ in range(self.n_values)]
        self._epoch_all_values = []
        self._all_values.append(self._epoch_all_values)
        self._epoch_all_times = []
        self._all_times.append(self._epoch_all_times)
        self.start_time = time.time()
        self.now_time = self.start_time
        self._make_epoch_text()

    def _bar_reset(self):
        """Reset attributes for progress bar."""
        self._bar_now_iter = 0
        self._bar_prop = 0.
        self._bar_values = [0 for _ in range(self.n_values)]
        self._bar_value_weights = [0 for _ in range(self.n_values)]
        self._now_bar_values = [0 for _ in range(self.n_values)]
        self._bar_note = self.note
        self._bar_start_time = time.time()
        self._bar_now_time = self.start_time

    def now_values(
        self,
        no_unpacked: bool = False
    ) -> Union[Number, List[Number]]:
        """
        Get current epoch values. Note that the value may differ from
        what is displayed on the progress bar.

        Args:
            no_unpacked (bool):
                If False, return a unpacked value if n_values=1.
                Defaults to False.
        Returns:
            Union[Number, List[Number]]:
                Current value if no_unpacked=True, else list of values.
        """
        if no_unpacked:
            return self._now_epoch_values
        else:
            if self.n_values == 1:
                return self._now_epoch_values[0]
            else:
                return self._now_epoch_values

    def _make_epoch_text(self):
        """Make epoch text."""
        if self.n_epochs:
            if self.unit >= 2:
                first = (self.n_bar - 1) * self.unit + 1
                last = self.n_bar * self.unit
                epoch_text = f"{first}-{last}"
                epoch_text = epoch_text.rjust(self._n_epoch_digits * 2 + 1)
            else:
                epoch_text = str(self.now_epoch).rjust(self._n_epoch_digits)
            epoch_text += f"/{self.n_epochs}"
        else:
            epoch_text = str(self.now_epoch)
        self._epoch_text = epoch_text

    def start(self, **kwargs):
        """
        Start running. Initialize start time and epoch. You can set the
        attributes to be used at this runtime. If not set, the default
        value is used. Arguemnts is the same as the constructor.

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
            momentum (float):
                Momentum for updating values. If None, the value is
                updated by the aggregation function. If set, the value
                is updated by the exponential moving average, agg_fn and
                weight are ignored. Defaults to None.
            width (int):
                Width of progress bar. Defaults to 40.
            leave_freq (int):
                Frequency of leaving the progress bar. If <= 0, none are
                left. Defaults to 1.
            unit (int):
                Unit of progress bar (epoch). Defaults to 1.
            defer (bool):
                If True, auto-step will be deferred until the next
                memo() call. Use when you want to update a note at the
                end of the epoch. Defaults to False.
            note (str):
                Note for progress bar. Defaults to ''.
            symbol (str):
                Symbol for progress bar. Defaults to '#'.
            round (int):
                Number of digits to round to. Default is 5.
            sep_label (str):
                Separator character for value and label.
                Defaults to ': '.
            sep_values (str):
                Separator character for values. Defaults to ', '.
            sep_note (str):
                Separator character for note. Defaults to ', '.
        """
        self.reset(**kwargs)
        assert self.n_iter is not None, "'n_iter' is not set."
        if ("label" not in kwargs) and (self.label is None) \
                and ("labels" in kwargs):
            warnings.warn(
                "'labels' is not a valid argument. "
                "Try using 'label' instead.",
            )
        self.is_running = True
        self.now_epoch = 1
        self.n_bar = 1
        self._keep_step = False
        self._bar_note = self.note
        self._make_epoch_text()
        self._tmp_time = self.now_time

    def _draw(self):
        """Draw progress bar."""
        index_text = f"{self._epoch_text}:"
        bar_text = self.symbol * int(self.width * self._bar_prop)
        bar_text = bar_text.ljust(self.width)
        prop_text = f"{int(self._bar_prop * 100)}%".rjust(4)
        bar_time = self._bar_now_time - self._bar_start_time
        time_text = f"[{time_format(bar_time)}]"
        value_texts = []
        for label, value in zip(self._labels, self._now_bar_values):
            value_text = label + self.sep_label if self.label else ""
            value_text += value_format(value, self.round)
            value_texts.append(value_text)
        value_text = self.sep_values.join(value_texts)
        text = " ".join([
            index_text,
            bar_text,
            prop_text,
            time_text,
            value_text,
        ])
        if self._bar_note:
            text += self.sep_note + self._bar_note
        print("\r" + " " * self._text_length, end="")
        print("\r" + text, end=" ", flush=True)
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
                if self.momentum:
                    self._now_bar_values[i] = (
                        self.momentum * self._now_bar_values[i]
                        + (1 - self.momentum) * value[i]
                    )
                    self._now_epoch_values[i] = (
                        self.momentum * self._now_epoch_values[i]
                        + (1 - self.momentum) * value[i]
                    )
                else:
                    if self._bar_value_weights[i]:
                        self._now_bar_values[i] = self._agg_fn(
                            self._bar_values[i], self._bar_value_weights[i]
                        )
                        self._now_epoch_values[i] = self._agg_fn(
                            self._epoch_values[i], self._epoch_value_weights[i]
                        )
                    else:
                        self._now_epoch_values[i] = 0.
                        self._now_bar_values[i] = 0.

        self.prop = self.now_iter / (self.n_iter * self.unit)
        self._bar_prop = self._bar_now_iter / (self.n_iter * self.unit)
        self.now_time = time.time()
        self._bar_now_time = time.time()

    def update(
        self,
        value: Optional[Union[Number, List[Number]]] = None,
        weight: Union[Number, List[Number]] = 1,
        advance: int = 1,
        auto_step: bool = True,
        note: Optional[str] = None,
    ):
        """
        Update progress bar and aggregate value.

        Args:
            value (Optional[Union[Number, List[Number]]]):
                value. If None, only the progress bar advances. Defaults
                to None.
            weight (Union[Number, List[Number]]):
                weight of value. Defaults to 1.
            advance (int):
                Number of iterations to advance. Defaults to 1.
            auto_step (bool):
                If True, step() is called when the number of iterations
                reaches n_iter. Defaults to True.
            note (Optional[str]):
                Note for progress bar. Defaults to None.
        """
        if not self.is_running:
            self.start()
        self._update_values(advance, value, weight)
        self.prop = self.now_iter / self.n_iter
        if note is not None:
            self._bar_note = note
        self._draw()

        if self.store_all_values:
            self._epoch_all_values.append(value)
        if self.store_all_times:
            self._epoch_all_times.append(self.now_time - self._tmp_time)

        self._tmp_time = self.now_time
        if auto_step and self.prop >= 1.:
            bar_step = self._bar_prop >= 1.
            leave = not (self.n_bar % self.leave_freq)
            leave = self.leave_freq > 0 and leave
            if self.defer:
                self._keep_step = True
                self._keep_step_info = {
                    "bar_step": bar_step,
                    "leave": leave,
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
                print("\n", end="")
            else:
                print("\r", " " * self._text_length, end="\r")
            self.n_bar += 1
            self._bar_reset()
        value = self._now_epoch_values
        if self.n_values == 1:
            value = value[0]
        self.values.append(value)
        self.now_epoch += 1
        self._epoch_reset()
        self._keep_step = False

    def _get_data(self, data: str, flatten: bool = False):
        if data == "value":
            if not self.store_all_values:
                warnings.warn(
                    "store_all_values is False. No values are stored."
                )
            data = self._all_values
        elif data == "time":
            if not self.store_all_times:
                warnings.warn(
                    "store_all_times is False. No times are stored."
                )
            data = self._all_times
        data = [epoch for epoch in data if epoch]
        if flatten:
            return [v for epoch in data for v in epoch]
        else:
            return data

    def get_all_values(
        self,
        flatten: bool = False
    ) -> Union[List[Numbers], List[List[Numbers]]]:
        """
        Get values of all iterations.

        Args:
            flatten (bool):
                If True, flatten to 1D sequence. else, 2D sequence of
                (epoch, value). Defaults to False.

        Returns:
            Union[List[Numbers], List[List[Numbers]]]:
                List[value] if flatten=True, else List[List[value]].
        """
        return self._get_data("value", flatten)

    def get_all_times(
        self,
        flatten: bool = False
    ) -> Union[List[Number], List[List[Number]]]:
        """
        Get times of all iterations.

        Args:
            flatten (bool):
                If True, flatten to 1D sequence. else, 2D sequence of
                (epoch, time). Defaults to False.

        Returns:
            List[Number]: List[time] if flatten=True, else
                List[List[time]].
        """
        return self._get_data("time", flatten)

    def memo(self, note: Optional[str] = None, no_step: bool = False):
        """
        Change note text for progress bar.

        Args:
            note (Optional[str]):
                Text. If None, the note is not changed. Defaults to
                None.
            no_step (bool):
                If True, step() is not called when be deferred. Defaults
                to False.
        """
        if note is not None:
            self._bar_note = note
        self._draw()
        if self._keep_step and not no_step:
            self.step(**self._keep_step_info)


def train_progress(with_test: bool = False, **kwargs) -> Progress:
    """
    Return a progress bar that is suited for machine learning training.

    Args:
        with_test (bool):
            Calculate the loss for the test data at the end of the
            epoch. If True, become defer=True for memo() at the end of
            the epoch, and other fine adjustments are made. Defaults to
            False.

    Returns:
        Progress: Progress bar object.
    """
    kwargs["label"] = kwargs.get("label", "loss")
    if with_test:
        kwargs["width"] = kwargs.get("width", 30)
        kwargs["defer"] = True
    prog = Progress(**kwargs)
    return prog

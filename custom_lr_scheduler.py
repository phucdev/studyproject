import types
from functools import partial

import warnings
import math

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class CustomLambdaLR(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False, min_lr=0.):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict


    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        lrs = []
        for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs):
            lr = base_lr * lmbda(self.last_epoch)   # self.last_epoch is essentially the number of steps
            lrs.append(max(lr, self.min_lr))
        return lrs


def _get_cosine_schedule_with_warmup_embedding_tuning_lr_lambda(
        current_step: int, *, warmup_percentage: int, num_training_steps: int, num_cycles: float,
        pure_embedding_training_percentage: int, embedding_tuning_warmup_percentage: int
):
    # TODO check if different warmup configurations are working as expected
    num_pure_embedding_training_steps = math.ceil(num_training_steps*pure_embedding_training_percentage/100)
    warmup_embedding_steps = math.ceil(num_pure_embedding_training_steps*embedding_tuning_warmup_percentage/100)
    warmup_and_embedding_training_steps = warmup_embedding_steps + num_pure_embedding_training_steps

    full_training_steps = num_training_steps - warmup_and_embedding_training_steps
    warmup_full_training_steps = math.ceil(full_training_steps*warmup_percentage/100)
    full_training_start_step = warmup_and_embedding_training_steps + warmup_full_training_steps

    # embedding training warmup phase
    if current_step < warmup_embedding_steps:
        return float(current_step) / float(max(1, warmup_embedding_steps))

    # pure embedding training phase
    elif warmup_embedding_steps <= current_step < warmup_and_embedding_training_steps:
        numerator = float(current_step - warmup_embedding_steps)
        denominator = float(max(1, num_pure_embedding_training_steps))
        progress = numerator / denominator

    # full training phase warmup phase
    elif warmup_and_embedding_training_steps <= current_step < full_training_start_step:
        return float(current_step - warmup_and_embedding_training_steps) / float(max(1, warmup_full_training_steps))

    # full training
    else:
        numerator = float(current_step - full_training_start_step)
        denominator = float(max(1, num_training_steps - full_training_start_step))
        progress = numerator / denominator
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress * float(num_cycles) * 2.0)))


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, warmup_percentage: int, num_training_steps: int, num_cycles: float):
    warmup_steps = math.ceil(num_training_steps*warmup_percentage/100)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress * float(num_cycles) * 2.0)))


def get_custom_lr_scheduler(optimizer, warmup_percentage, num_training_steps, pure_embedding_training_percentage,
                            num_cycles: float = 0.5, min_lr: float = 0.0, embedding_tuning_warmup_percentage: int = 0):
    if pure_embedding_training_percentage > 0:
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_embedding_tuning_lr_lambda,
            warmup_percentage=warmup_percentage,
            num_training_steps=num_training_steps,
            pure_embedding_training_percentage=pure_embedding_training_percentage,
            num_cycles=num_cycles,
            embedding_tuning_warmup_percentage=embedding_tuning_warmup_percentage
        )
    else:
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            warmup_percentage=warmup_percentage,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )

    return CustomLambdaLR(optimizer, lr_lambda, min_lr=min_lr)

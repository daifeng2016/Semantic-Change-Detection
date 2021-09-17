import types
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math


def linear(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return last_epoch * lr_goal / warm_epochs


def quad_square(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return math.sqrt(last_epoch) * lr_goal / math.sqrt(warm_epochs)


def quadratic(last_epoch, lr_goal, warm_epochs):
    if last_epoch > warm_epochs:
        return -1
    return (last_epoch ** 2) * lr_goal / (warm_epochs ** 2)







class FuncLRScheduler(_LRScheduler):

    def __init__(self, optimizer: Optimizer, lr_goal, warm_epochs, scheduler_after,
                 last_epoch: int = -1, fns=None):
        self.optimizer = optimizer
        self.func = []
        self.warm_epochs = []
        self.next_scheduler = []
        self.lr_goal = []
        self.last_epoch = last_epoch
        self.last_epochs = [0] * len(warm_epochs)

        if not isinstance(fns, list) and not isinstance(fns, tuple):
            self.func = [fns] * len(optimizer.param_groups)
        for item in fns:
            if isinstance(item, types.FunctionType):
                self.func.append(item)
            else:
                raise ValueError("Expected fn to be a function, but got {}".format(type(item)))
        if len(fns) != len(optimizer.param_groups):
            raise ValueError("Expected {} functions, but got {}".format(
                len(optimizer.param_groups), len(fns)))

        if not isinstance(warm_epochs, list) and not isinstance(warm_epochs, tuple):
            self.warm_epochs = [warm_epochs] * len(optimizer.param_groups)
        for item in warm_epochs:
            if isinstance(item, int):
                self.warm_epochs.append(item)
            else:
                raise ValueError("Expected warm_epochs to be an int, but got {}".format(type(item)))
        if len(warm_epochs) != len(optimizer.param_groups):
            raise ValueError("Expected {} warm_epochs, but got {}".format(
                len(optimizer.param_groups), len(warm_epochs)))

        if not isinstance(lr_goal, list) and not isinstance(lr_goal, tuple):
            self.lr_goal = [lr_goal] * len(optimizer.param_groups)
        for item in lr_goal:
            if isinstance(item, float) | isinstance(item, int):
                self.lr_goal.append(item)
            else:
                raise ValueError("Expected learning rate to be float or int, but got {}".format(type(item)))
        else:
            if len(lr_goal) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_goal)))
            self.lr_goal = list(lr_goal)

        if not isinstance(scheduler_after, list) and not isinstance(scheduler_after, tuple):
            self.next_scheduler = [scheduler_after] * len(optimizer.param_groups)
        for item in scheduler_after:
            if isinstance(item, _LRScheduler):
                self.next_scheduler.append(item)
            else:
                raise ValueError("Expected scheduler_after to be a scheduler, but got {}".format(type(item)))
        if len(scheduler_after) != len(optimizer.param_groups):
            raise ValueError("Expected {} scheduler_after, but got {}".format(
                len(optimizer.param_groups), len(scheduler_after)))

        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if (key != 'optimizer') & (key != 'next_scheduler')}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def get_lr(self):
        to_return = [func(self.last_epoch, lr_goal, warm_epochs) for func, lr_goal, warm_epochs in
                     zip(self.func, self.lr_goal, self.warm_epochs)]
        for index, item in enumerate(to_return):
            if item < 0:
                self.next_scheduler[index].last_epoch = self.last_epochs[index] + 1
                to_return[index] = self.next_scheduler[index].get_lr()[index]
                self.last_epochs[index] = self.next_scheduler[index].last_epoch
        return to_return

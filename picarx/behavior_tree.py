from __future__ import annotations

from enum import Enum, auto
from typing import Callable, List, Optional


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class Node:
    def tick(self) -> Status:
        raise NotImplementedError


class Action(Node):
    def __init__(self, func: Callable[[], Status]):
        self.func = func

    def tick(self) -> Status:
        return self.func()


class Condition(Node):
    def __init__(self, predicate: Callable[[], bool]):
        self.predicate = predicate

    def tick(self) -> Status:
        return Status.SUCCESS if self.predicate() else Status.FAILURE


class Sequence(Node):
    def __init__(self, children: List[Node]):
        self.children = children

    def tick(self) -> Status:
        for child in self.children:
            status = child.tick()
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


class Selector(Node):
    def __init__(self, children: List[Node]):
        self.children = children

    def tick(self) -> Status:
        for child in self.children:
            status = child.tick()
            if status != Status.FAILURE:
                return status
        return Status.FAILURE

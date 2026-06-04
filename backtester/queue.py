from collections import deque


class EventQueue:
    def __init__(self):
        self._q = deque()

    def put(self, event) -> None:
        self._q.append(event)

    def get(self):
        return self._q.popleft()

    def empty(self) -> bool:
        return len(self._q) == 0

    def __len__(self) -> int:
        return len(self._q)

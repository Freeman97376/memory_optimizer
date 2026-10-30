"""Single-process admission for the initial one-worker validation deployment."""

from threading import Lock

from memory_optimizer.domain import ProcessingBusy


class SingleAdd:
    def __init__(self, delegate):
        self.delegate = delegate
        self._gate = Lock()

    def add(self, command):
        if not self._gate.acquire(blocking=False):
            raise ProcessingBusy("Add capacity reached")
        try:
            return self.delegate.add(command)
        finally:
            self._gate.release()

class RuntimeNotReadyError(RuntimeError):
    """The runtime cannot currently accept model operations."""


class OperationWaitTimeoutError(TimeoutError):
    """An operation exhausted its wait for model access."""


class RuntimeBusyError(RuntimeError):
    """The operation's waiting queue has reached capacity."""
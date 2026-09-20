from enum import Enum, auto
from dataclasses import dataclass

class OperationKind(Enum):
    CLASSIFICATION = auto()
    EXPLANATION = auto()

@dataclass(frozen=True, eq=False)
class WaitingOperation:
    kind: OperationKind
    deadline: float

def choose_next_kind(
        *,
        classification_waiting: bool,
        explanation_waiting: bool,
        consecutive_classifications: int,
        classification_burst_limit: int,
) -> OperationKind | None:
    """
    pure function with the role of selection rule
    """
    match (classification_waiting, explanation_waiting):
        case (False, False):
            return
        case (True, False):
            return OperationKind.CLASSIFICATION
        case (False, True):
            return OperationKind.EXPLANATION
        case (True, True):
            return OperationKind.CLASSIFICATION if consecutive_classifications < classification_burst_limit else OperationKind.EXPLANATION

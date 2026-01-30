from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ClarificationQuestion:
    slot: str
    question: str
    options: List[str]
    priority: int = 0


@dataclass
class TriageResult:
    status: str  # "READY" | "NEED_CLARIFICATION"
    assumptions_allowed: bool
    extracted: Dict[str, Any]
    missing_slots: List[str]
    questions: List[ClarificationQuestion]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["questions"] = [asdict(q) for q in self.questions]
        return d

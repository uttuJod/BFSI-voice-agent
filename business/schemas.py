from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    ERROR = "error"


class ToolExecutionResult(BaseModel):
    tool: str

    status: ToolExecutionStatus

    success: bool = False

    needs_clarification: bool = False

    user_message: str

    data: dict[str, Any] = Field(
        default_factory=dict
    )

    error_code: str | None = None

    @classmethod
    def ok(
        cls,
        tool: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> "ToolExecutionResult":

        return cls(
            tool=tool,
            status=ToolExecutionStatus.SUCCESS,
            success=True,
            needs_clarification=False,
            user_message=message,
            data=data or {},
        )

    @classmethod
    def clarify(
        cls,
        tool: str,
        message: str,
        error_code: str | None = None,
    ) -> "ToolExecutionResult":

        return cls(
            tool=tool,
            status=(
                ToolExecutionStatus.NEEDS_CLARIFICATION
            ),
            success=False,
            needs_clarification=True,
            user_message=message,
            error_code=error_code,
        )

    @classmethod
    def blocked(
        cls,
        tool: str,
        message: str,
        error_code: str | None = None,
    ) -> "ToolExecutionResult":

        return cls(
            tool=tool,
            status=ToolExecutionStatus.BLOCKED,
            success=False,
            needs_clarification=False,
            user_message=message,
            error_code=error_code,
        )

    @classmethod
    def error(
        cls,
        tool: str,
        message: str,
        error_code: str | None = None,
    ) -> "ToolExecutionResult":

        return cls(
            tool=tool,
            status=ToolExecutionStatus.ERROR,
            success=False,
            needs_clarification=False,
            user_message=message,
            error_code=error_code,
        )
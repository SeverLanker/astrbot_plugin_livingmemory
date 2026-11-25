"""按消息角色划分反思窗口里的用户证据。

提取只使用对方真正发出的消息。助手角色、机器人标记、主动对话标记，
以及主动对话插件写进会话的任务模板，都不算用户证据。窗口里没有
用户证据时，调用方不写记忆，只推进总结游标。

这里不做措辞审查。用户原话的改写可以保留；模型若仍凭空生成原文里
没有的状态、关系或失联，那是模型行为，不在这一层用关键词删除。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SYNTHETIC_TASK_RE = re.compile(
    r"\[系统任务[：:]|"
    r"你被授权在(?:私聊|群聊)中发起|"
    r"\{\{current_time\}\}.*\{\{unanswered_count\}\}|"
    r"\{\{unanswered_count\}\}.*\{\{current_time\}\}",
    re.DOTALL,
)
_MEDIA_ONLY = {"[图片消息]", "[图片]", "[语音]", "[视频]", "[文件]", "[表情]"}
_ASSISTANT_ROLES = {"assistant", "bot"}
_NON_USER_ROLES = {"system", "tool", "function"}


@dataclass(slots=True)
class WindowEvidence:
    """一个总结窗口里，角色上可当作证据的用户消息统计。"""

    user_count: int = 0
    assistant_count: int = 0
    session_ids: list[str] = field(default_factory=list)

    @property
    def lacks_user_evidence(self) -> bool:
        return self.user_count == 0


def blocks_user_fact_storage(metadata: dict[str, Any] | None) -> bool:
    """没有用户证据的结果不得写入用户事实、原子或图谱。"""
    if not isinstance(metadata, dict):
        return False
    attribution = metadata.get("source_attribution")
    if not isinstance(attribution, dict):
        return False
    if attribution.get("assistant_only") is True:
        return True
    return attribution.get("store_user_facts") is False


def select_user_evidence(messages: list[Any]) -> tuple[WindowEvidence, list[Any]]:
    """返回用户证据消息，顺序与原窗口一致。

    助手消息仍留在原会话里，只是不进入这次事实提取的输入。
    """
    evidence = WindowEvidence()
    selected: list[Any] = []
    for message in messages or []:
        text = _message_text(message).strip()
        if _is_assistant_like(message, text):
            if text:
                evidence.assistant_count += 1
            continue
        role = _message_role(message)
        if role in _NON_USER_ROLES:
            continue
        if not text or text in _MEDIA_ONLY:
            continue
        evidence.user_count += 1
        session_id = _message_session_id(message)
        if session_id and session_id not in evidence.session_ids:
            evidence.session_ids.append(session_id)
        selected.append(message)
    return evidence, selected


def evidence_record(evidence: WindowEvidence) -> dict[str, Any]:
    """记录这条记忆的证据来自哪些会话里的用户消息。"""
    return {
        "assistant_only": evidence.lacks_user_evidence,
        "store_user_facts": not evidence.lacks_user_evidence,
        "user_message_count": evidence.user_count,
        "assistant_message_count": evidence.assistant_count,
        "evidence_session_ids": list(evidence.session_ids),
    }


def empty_unattributed_metadata(
    *,
    is_group_chat: bool,
    evidence: WindowEvidence,
) -> dict[str, Any]:
    """构造一个不会进入原子或图谱的空结果。"""
    return {
        "topics": [],
        "key_facts": [],
        "sentiment": "neutral",
        "interaction_type": "group_chat" if is_group_chat else "private_chat",
        "canonical_summary": "",
        "persona_summary": "",
        "summary_schema_version": "v2",
        "summary_quality": "skipped",
        "source_attribution": evidence_record(evidence),
    }


def _is_assistant_like(message: Any, text: str) -> bool:
    role = _message_role(message)
    if role in _ASSISTANT_ROLES:
        return True
    metadata = _message_metadata(message)
    if metadata.get("is_bot_message") is True:
        return True
    if metadata.get("proactive") is True or metadata.get("is_proactive") is True:
        return True
    return bool(text) and _SYNTHETIC_TASK_RE.search(text) is not None


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").strip().lower()
    return str(getattr(message, "role", "") or "").strip().lower()


def _message_metadata(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        metadata = message.get("metadata") or {}
    else:
        metadata = getattr(message, "metadata", None) or {}
    return metadata if isinstance(metadata, dict) else {}


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    from ..models.conversation_models import Message

    return Message.content_to_text(content)


def _message_session_id(message: Any) -> str:
    if isinstance(message, dict):
        value = message.get("session_id")
    else:
        value = getattr(message, "session_id", None)
    return str(value or "").strip()


__all__ = [
    "WindowEvidence",
    "blocks_user_fact_storage",
    "empty_unattributed_metadata",
    "evidence_record",
    "select_user_evidence",
]

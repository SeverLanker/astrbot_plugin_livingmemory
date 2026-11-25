"""把反思窗口里的发言角色和可写入的用户事实分开。

助手的主动问候、推测和提问不能证明用户状态、关系或失联。
只有窗口里的真实用户发言能支撑这类事实；某一渠道没有消息，
也不能写成“任何平台都没有联系”，时长也不能来自助手原文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_ABSENCE_RE = re.compile(
    r"没(?:有)?联系|未联系|不联系|失联|"
    r"好久没|很久没|许久没|好久不见|"
    r"没(?:有)?找我|没来找|不来找|"
    r"没理我|不理我|没回我|没有回复|未回复|不回复|"
    r"没(?:有)?消息|没(?:有)?说话|"
    r"没(?:有)?(?:怎么)?上(?:线|QQ|qq|微信)|未上线|"
    r"失踪|"
    r"hasn'?t (?:contacted|spoken|messaged|replied|talked)|"
    r"have not contacted|no contact|gone silent",
    re.IGNORECASE,
)
_RELATION_STATUS_RE = re.compile(
    r"关系(?:变|疏|淡|冷|好了|差|破裂|亲近|疏远|紧张|结束)|"
    r"疏远|冷落|变淡|把我忘|忘记我|忘了我|不想理|"
    r"感情(?:变|破裂|淡了|变淡)|"
    r"relationship (?:changed|faded|ended|cold)|forgot me|ignoring me|drifted",
    re.IGNORECASE,
)
_RELATION_ROLE_RE = re.compile(r"同事|朋友|同学|家人|亲戚|情侣|夫妻|老板|室友|搭档")
_USER_STATE_RE = re.compile(
    r"心情(?:很|特别|比较|有点|不太|非常)?(?:好|差|不好|低落|糟糕|烦|坏)|"
    r"情绪(?:低落|不好|崩溃)|"
    r"很难过|很生气|在生气|很焦虑|压力很大|"
    r"(?:很|太|正)忙|在忙|"
    r"身体不舒服|生病了|"
    r"很孤独|很寂寞|想我了|不想理我"
)
_GLOBAL_RE = re.compile(
    r"任何(?:地方|平台|渠道|聊天|会话|端)|"
    r"所有(?:平台|渠道|聊天|会话|端)|"
    r"各个平台|全平台|哪里都|哪儿都|"
    r"无论哪|不管在哪|"
    r"都没(?:有)?联系|从未联系|"
    r"QQ还是微信|微信还是QQ|不管是QQ|不论QQ|无论QQ|"
    r"QQ和微信都|微信和QQ都|"
    r"anywhere|any platform|all platforms|all channels|every platform",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\d+\s*(?:天|日|周|星期|个月)|"
    r"[零一二三四五六七八九十两几数多]+\s*(?:天|日|周|星期|个月)|"
    r"好几[天日周]|好几个月|数周|数月|很久|好久|许久|"
    r"这周|本周|上周|下周|"
    r"\d+\s*(?:days?|weeks?|months?)",
    re.IGNORECASE,
)
_SYNTHETIC_TASK_RE = re.compile(
    r"\[系统任务[：:]|"
    r"你被授权在(?:私聊|群聊)中发起|"
    r"\{\{current_time\}\}.*\{\{unanswered_count\}\}|"
    r"\{\{unanswered_count\}\}.*\{\{current_time\}\}",
    re.DOTALL,
)
_MEDIA_ONLY = {"[图片消息]", "[图片]", "[语音]", "[视频]", "[文件]", "[表情]"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


@dataclass(slots=True)
class WindowEvidence:
    """一个总结窗口里可当作证据的用户原文。"""

    user_texts: list[str] = field(default_factory=list)
    user_labels: list[str] = field(default_factory=list)
    user_count: int = 0
    assistant_count: int = 0

    @property
    def user_blob(self) -> str:
        return "\n".join(self.user_texts)

    @property
    def lacks_user_evidence(self) -> bool:
        return self.user_count == 0


def blocks_user_fact_storage(metadata: dict[str, Any] | None) -> bool:
    """该结果是否禁止写入用户、关系或失联事实。"""
    if not isinstance(metadata, dict):
        return False
    attribution = metadata.get("source_attribution")
    if not isinstance(attribution, dict):
        return False
    if attribution.get("assistant_only") is True:
        return True
    return attribution.get("store_user_facts") is False


def collect_window_evidence(messages: list[Any]) -> WindowEvidence:
    """统计真实用户发言；助手消息和主动对话任务模板都不算用户证据。"""
    evidence = WindowEvidence()
    for message in messages or []:
        text = _message_text(message).strip()
        if _is_assistant_like(message, text):
            if text:
                evidence.assistant_count += 1
            continue
        role = _message_role(message)
        if role in {"system", "tool", "function"}:
            continue
        if not text or text in _MEDIA_ONLY:
            continue
        evidence.user_count += 1
        evidence.user_texts.append(text)
        evidence.user_labels.append(_sender_label(message))
    return evidence


def empty_unattributed_metadata(
    *,
    is_group_chat: bool,
    evidence: WindowEvidence,
    reason: str,
    dropped_ungrounded_claims: int = 0,
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
        "source_attribution": {
            "assistant_only": evidence.lacks_user_evidence,
            "store_user_facts": False,
            "user_message_count": evidence.user_count,
            "assistant_message_count": evidence.assistant_count,
            "dropped_ungrounded_claims": dropped_ungrounded_claims,
            "reason": reason,
        },
    }


def apply_source_attribution(
    structured: dict[str, Any], evidence: WindowEvidence
) -> tuple[dict[str, Any], dict[str, Any]]:
    """删掉没有用户原文支撑的状态、关系、失联和时长说法。

    模型已经抽出、并且确实能在用户原文里找到的事实会保留。
    如果模型只返回了助手推测，则补上用户原话摘录，避免把真实发言丢掉。
    """
    data = dict(structured)
    user_blob = evidence.user_blob
    dropped = 0

    summary, summary_dropped = _scrub_prose(str(data.get("summary") or ""), user_blob)
    dropped += summary_dropped
    data["summary"] = summary

    canonical = data.get("canonical_summary")
    if isinstance(canonical, str) and canonical.strip():
        cleaned, canonical_dropped = _scrub_prose(canonical, user_blob)
        dropped += canonical_dropped
        data["canonical_summary"] = cleaned

    kept_facts: list[str] = []
    for fact in data.get("key_facts") or []:
        text = str(fact).strip()
        if not text:
            continue
        if claim_is_ungrounded(text, user_blob, include_roles=True):
            dropped += 1
            continue
        kept_facts.append(text)

    kept_topics: list[str] = []
    for topic in data.get("topics") or []:
        text = str(topic).strip()
        if not text:
            continue
        if claim_is_ungrounded(text, user_blob, include_roles=False):
            dropped += 1
            continue
        kept_topics.append(text)

    if dropped and not _user_content_covered(evidence, summary, kept_facts):
        for excerpt in _user_excerpts(evidence):
            if excerpt not in kept_facts:
                kept_facts.append(excerpt)
            if len(kept_facts) >= 5:
                break

    data["key_facts"] = kept_facts[:5]
    data["topics"] = kept_topics[:5]
    if dropped:
        data["importance"] = min(_as_float(data.get("importance"), 0.5), 0.5)

    store = bool(str(data.get("summary") or "").strip() or data["key_facts"])
    attribution = {
        "assistant_only": False,
        "store_user_facts": store,
        "user_message_count": evidence.user_count,
        "assistant_message_count": evidence.assistant_count,
        "dropped_ungrounded_claims": dropped,
        "reason": "filtered" if dropped else "grounded",
    }
    return data, attribution


def claim_is_ungrounded(
    text: str, user_blob: str, *, include_roles: bool
) -> bool:
    """判断一句话是不是缺少用户证据的状态、关系、失联或时长。"""
    if not text or not str(text).strip():
        return False
    if _pattern_ungrounded(_ABSENCE_RE, text, user_blob):
        return True
    if _duration_ungrounded(text, user_blob):
        return True
    if _pattern_ungrounded(_RELATION_STATUS_RE, text, user_blob):
        return True
    if _pattern_ungrounded(_USER_STATE_RE, text, user_blob):
        return True
    if _pattern_ungrounded(_GLOBAL_RE, text, user_blob):
        return True
    if include_roles and _pattern_ungrounded(_RELATION_ROLE_RE, text, user_blob):
        return True
    return False


def _scrub_prose(text: str, user_blob: str) -> tuple[str, int]:
    raw = str(text or "")
    if not raw.strip():
        return raw, 0
    if not claim_is_ungrounded(raw, user_blob, include_roles=False):
        return raw, 0
    parts = [
        part.strip()
        for part in _SENTENCE_SPLIT_RE.split(raw)
        if part and part.strip()
    ]
    if len(parts) <= 1:
        return "", 1
    kept: list[str] = []
    dropped = 0
    for part in parts:
        if claim_is_ungrounded(part, user_blob, include_roles=False):
            dropped += 1
            continue
        kept.append(part)
    if not kept:
        return "", dropped or 1
    return "".join(kept).strip(), dropped


def _pattern_ungrounded(pattern: re.Pattern[str], text: str, user_blob: str) -> bool:
    terms = [match.group(0) for match in pattern.finditer(text)]
    if not terms:
        return False
    user_norm = _norm(user_blob)
    return any(_norm(term) not in user_norm for term in terms)


def _duration_ungrounded(text: str, user_blob: str) -> bool:
    """相对时长必须出现在用户原文里，不能取自助手的推测。"""
    user_norm = _norm(user_blob)
    for match in _DURATION_RE.finditer(text):
        if _norm(match.group(0)) not in user_norm:
            return True
    return False


def _user_content_covered(
    evidence: WindowEvidence, summary: str, facts: list[str]
) -> bool:
    corpus = _norm("\n".join([summary, *facts]))
    if not evidence.user_texts:
        return True
    for text in evidence.user_texts:
        if not _text_covered(text, corpus):
            return False
    return True


def _text_covered(text: str, corpus_norm: str) -> bool:
    compact = _norm(text)
    grams = [
        compact[index : index + 2]
        for index in range(max(0, len(compact) - 1))
    ]
    grams = [gram for gram in grams if _meaningful_gram(gram)]
    if not grams:
        return True
    hits = sum(1 for gram in grams if gram in corpus_norm)
    return hits >= max(1, len(grams) // 4)


def _meaningful_gram(gram: str) -> bool:
    return not all(char in "的了呢吧啊我你是在有和也就都很把被这那" for char in gram)


def _user_excerpts(evidence: WindowEvidence) -> list[str]:
    excerpts: list[str] = []
    for label, text in zip(evidence.user_labels, evidence.user_texts):
        snippet = " ".join(text.split())
        if len(snippet) > 80:
            snippet = snippet[:80] + "…"
        excerpts.append(f"{label}：{snippet}")
    return excerpts


def _is_assistant_like(message: Any, text: str) -> bool:
    role = _message_role(message)
    if role == "assistant":
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


def _sender_label(message: Any) -> str:
    if isinstance(message, dict):
        name = message.get("sender_name") or message.get("sender_id")
    else:
        name = getattr(message, "sender_name", None) or getattr(
            message, "sender_id", None
        )
    label = str(name or "").strip()
    return label or "对方"


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "WindowEvidence",
    "apply_source_attribution",
    "blocks_user_fact_storage",
    "claim_is_ungrounded",
    "collect_window_evidence",
    "empty_unattributed_metadata",
]

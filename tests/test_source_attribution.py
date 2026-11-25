"""助手发言不能变成用户状态、关系或失联事实。"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from astrbot_plugin_livingmemory.core.base.config_manager import ConfigManager
from astrbot_plugin_livingmemory.core.command_handler import CommandHandler
from astrbot_plugin_livingmemory.core.models.conversation_models import Message
from astrbot_plugin_livingmemory.core.models.memory_atom import AtomType, MemoryAtom
from astrbot_plugin_livingmemory.core.page_api_modules.memory_handler import MemoryHandler
from astrbot_plugin_livingmemory.core.page_api_modules.utils import PageApiUtils
from astrbot_plugin_livingmemory.core.processors.graph_extractor import GraphExtractor
from astrbot_plugin_livingmemory.core.processors.memory_processor import MemoryProcessor
from astrbot_plugin_livingmemory.core.processors.source_attribution import (
    claim_is_ungrounded,
)

from tests.event_handler_support import _make_real_handler

QQ_SESSION = "aiocqhttp:FriendMessage:qq:10001"


class _DummyLLMProvider:
    def __init__(self, completion_text: str):
        self._completion_text = completion_text
        self.text_chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, prompt: str, system_prompt: str):
        return SimpleNamespace(completion_text=self._completion_text)


def _message(
    *,
    message_id: int,
    role: str,
    content: str,
    sender_id: str,
    sender_name: str,
    timestamp: float,
    bot: bool = False,
) -> Message:
    return Message(
        id=message_id,
        session_id=QQ_SESSION,
        role=role,
        content=content,
        sender_id=sender_id,
        sender_name=sender_name,
        group_id=None,
        platform="aiocqhttp",
        timestamp=timestamp,
        metadata={"is_bot_message": True} if bot else {},
    )


def _assistant_only_messages() -> list[Message]:
    base = time.time() - 10 * 86400
    texts = [
        "早安呀，好久没见你了，我有点想你。",
        "你是不是好几周都没来找我？我们的关系是不是变淡了？",
        "不管在QQ还是微信，你是不是哪里都不在了？",
    ]
    return [
        _message(
            message_id=index + 1,
            role="assistant",
            content=text,
            sender_id="bot",
            sender_name="Bot",
            timestamp=base + index * 3600,
            bot=True,
        )
        for index, text in enumerate(texts)
    ]


def _poisoned_summary() -> str:
    return json.dumps(
        {
            "summary": "主人已经好几周没有在任何平台联系我，我们的关系变淡了。",
            "topics": ["失联", "关系"],
            "key_facts": [
                "主人已经好几周没有在任何平台联系过我",
                "我们的关系变淡了",
                "主人心情很差",
            ],
            "sentiment": "negative",
            "importance": 0.99,
        },
        ensure_ascii=False,
    )


def _graph_text(metadata: dict, atoms: list) -> str:
    graph = GraphExtractor().extract(
        1,
        metadata.get("canonical_summary") or metadata.get("persona_summary") or "",
        metadata,
        atoms,
    )
    parts = [entry.content for entry in graph.entries]
    parts.extend(node.value for node in graph.nodes)
    parts.extend(atom.content for atom in atoms)
    return "\n".join(parts)


def _assert_no_false_user_facts(blob: str) -> None:
    for phrase in (
        "任何平台",
        "所有平台",
        "好几周",
        "关系变淡",
        "哪里都",
        "QQ还是微信",
    ):
        assert phrase not in blob


@pytest.mark.asyncio
async def test_assistant_only_window_skips_llm_and_creates_no_facts():
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        _assistant_only_messages()
    )

    llm.text_chat.assert_not_awaited()
    assert content == ""
    assert importance == 0.0
    assert metadata["key_facts"] == []
    assert metadata["source_attribution"]["store_user_facts"] is False
    assert metadata["source_attribution"]["assistant_only"] is True
    atoms = processor.classify_atoms_from_metadata(metadata)
    assert atoms == []
    assert _graph_text(metadata, atoms) == ""


@pytest.mark.asyncio
async def test_synthetic_proactive_prompt_is_not_user_evidence():
    base = time.time() - 86400
    messages = [
        _message(
            message_id=1,
            role="user",
            content=(
                "[系统任务：主动对话]\n"
                "我们好像已经有一段时间没有说话了，我应该主动打破沉默。\n"
                "未回复次数大于 0 时可以带一点失落。"
            ),
            sender_id="10001",
            sender_name="主人",
            timestamp=base,
        ),
        *_assistant_only_messages(),
    ]
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _content, metadata, _importance = await processor.process_conversation(messages)

    llm.text_chat.assert_not_awaited()
    assert metadata["source_attribution"]["store_user_facts"] is False
    assert processor.classify_atoms_from_metadata(metadata) == []


@pytest.mark.asyncio
async def test_mixed_window_keeps_user_facts_and_drops_assistant_speculation():
    base = time.time() - 40 * 86400
    messages = [
        _message(
            message_id=1,
            role="user",
            content="我今天吃了牛肉面，李四是我同事，下周一起开会。我今天心情很差。",
            sender_id="10001",
            sender_name="主人",
            timestamp=base,
        ),
        _message(
            message_id=2,
            role="assistant",
            content="你是不是好几周都没来找我？不管QQ还是微信，关系是不是变淡了？",
            sender_id="bot",
            sender_name="Bot",
            timestamp=base + 30 * 86400,
            bot=True,
        ),
    ]
    llm = _DummyLLMProvider(
        json.dumps(
            {
                "summary": (
                    "主人今天吃了牛肉面。"
                    "我猜主人已经好几周没有在任何平台联系我，关系也变淡了。"
                ),
                "topics": ["饮食", "失联"],
                "key_facts": [
                    "主人今天吃了牛肉面",
                    "李四是主人的同事",
                    "下周一起开会",
                    "主人今天心情很差",
                    "主人已经好几周没有在任何平台联系过我",
                    "我们的关系变淡了",
                ],
                "sentiment": "negative",
                "importance": 0.95,
            },
            ensure_ascii=False,
        )
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(messages)

    llm.text_chat.assert_awaited()
    assert metadata["source_attribution"]["store_user_facts"] is True
    assert metadata["source_attribution"]["dropped_ungrounded_claims"] > 0
    assert "牛肉面" in content
    assert "同事" in content
    assert "下周" in content
    assert "心情很差" in content
    _assert_no_false_user_facts(content)
    assert "失联" not in metadata["topics"]
    assert importance == 0.5
    assert "absence" not in metadata
    assert "silence_days" not in metadata
    atoms = processor.classify_atoms_from_metadata(
        metadata, session_id=QQ_SESSION, persona_id="persona-1"
    )
    blob = content + "\n" + _graph_text(metadata, atoms)
    assert "牛肉面" in blob
    _assert_no_false_user_facts(blob)
    assert all(atom.session_id == QQ_SESSION for atom in atoms)
    assert AtomType.RELATIONAL not in {atom.atom_type for atom in atoms if "任何" in atom.content}


def test_channel_specific_absence_stays_channel_specific():
    user_blob = "我这周没怎么上QQ，但微信一直在聊"
    assert claim_is_ungrounded("主人这周没怎么上QQ", user_blob, include_roles=True) is False
    assert (
        claim_is_ungrounded(
            "主人好几周没有在任何平台联系过我", user_blob, include_roles=True
        )
        is True
    )
    assert (
        claim_is_ungrounded("不管QQ还是微信都失联了", user_blob, include_roles=True)
        is True
    )


def test_duration_and_question_from_assistant_are_not_evidence():
    user_blob = "我今天吃了牛肉面"
    assert claim_is_ungrounded("你是不是好几周没来找我？", user_blob, include_roles=True)
    assert claim_is_ungrounded("主人心情很差", user_blob, include_roles=True)
    assert not claim_is_ungrounded("主人今天吃了牛肉面", user_blob, include_roles=True)


def test_graph_blocks_poisoned_atoms_only_when_attribution_says_so():
    poisoned = MemoryAtom(
        parent_memory_id=0,
        content="主人已经好几周没有在任何平台联系过我",
        atom_type=AtomType.RELATIONAL,
        session_id=QQ_SESSION,
    )
    blocked = {
        "key_facts": [poisoned.content],
        "canonical_summary": poisoned.content,
        "source_attribution": {
            "assistant_only": True,
            "store_user_facts": False,
        },
    }
    blocked_graph = GraphExtractor().extract(7, poisoned.content, blocked, [poisoned])
    assert blocked_graph.entries == []
    assert blocked_graph.nodes == []

    historical = {"key_facts": ["主人这周没怎么上QQ"], "canonical_summary": "原有记忆"}
    historical_graph = GraphExtractor().extract(8, "原有记忆", historical, None)
    assert any("QQ" in entry.content for entry in historical_graph.entries)


@pytest.mark.asyncio
async def test_reflection_assistant_only_advances_cursor_without_loop(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    base = time.time() - 5 * 86400
    for index, text in enumerate(
        [
            "早安呀，好久没见你了。",
            "你是不是好几周都没来找我？",
            "不管QQ还是微信，你是不是把我忘了？",
        ]
    ):
        await manager.add_message(
            session_id=session_id,
            role="assistant",
            content=text,
            sender_id="bot",
            sender_name="Bot",
            platform="aiocqhttp",
            is_bot_message=True,
            timestamp=base + index * 3600,
        )
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)
    handler._memory_reflection.memory_processor = processor
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)
    messages = await manager.get_messages_range(
        session_id=session_id, start_index=0, end_index=3
    )
    try:
        committed = await handler._memory_reflection._storage_task(
            session_id,
            messages,
            "persona-1",
            0,
            3,
            0,
            memory_scope=session_id,
        )
        assert committed is True
        llm.text_chat.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 3
        assert await manager.get_session_metadata(session_id, "pending_summary", None) is None

        again = await handler._memory_reflection._storage_task(
            session_id,
            messages,
            "persona-1",
            0,
            3,
            0,
            memory_scope=session_id,
        )
        assert again is False
        llm.text_chat.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_reflection_mixed_window_stores_session_scoped_user_fact(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    base = time.time() - 86400
    await manager.add_message(
        session_id=session_id,
        role="user",
        content="我今天吃了牛肉面",
        sender_id="10001",
        sender_name="主人",
        platform="aiocqhttp",
        timestamp=base,
    )
    await manager.add_message(
        session_id=session_id,
        role="assistant",
        content="你是不是好几周没在任何平台联系我？",
        sender_id="bot",
        sender_name="Bot",
        platform="aiocqhttp",
        is_bot_message=True,
        timestamp=base + 60,
    )
    llm = _DummyLLMProvider(
        json.dumps(
            {
                "summary": "主人今天吃了牛肉面。另外我怀疑他好几周没有在任何平台联系我。",
                "topics": ["饮食"],
                "key_facts": [
                    "主人今天吃了牛肉面",
                    "主人已经好几周没有在任何平台联系过我",
                ],
                "sentiment": "neutral",
                "importance": 0.9,
            },
            ensure_ascii=False,
        )
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)
    handler._memory_reflection.memory_processor = processor
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=11)
    messages = await manager.get_messages_range(
        session_id=session_id, start_index=0, end_index=2
    )
    try:
        committed = await handler._memory_reflection._storage_task(
            session_id,
            messages,
            "persona-1",
            0,
            2,
            0,
            memory_scope=session_id,
        )
        assert committed is True
        handler.memory_engine.add_memory.assert_awaited_once()
        stored = handler.memory_engine.add_memory.await_args.kwargs
        assert stored["session_id"] == QQ_SESSION
        assert stored["session_id"] != "livingmemory:global"
        assert "牛肉面" in stored["content"]
        _assert_no_false_user_facts(stored["content"])
        assert all("任何平台" not in atom.content for atom in stored["atoms"])
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 2
    finally:
        await handler.shutdown()
        await store.close()


def _session_event():
    return SimpleNamespace(
        unified_msg_origin=QQ_SESSION,
        plain_result=lambda message: message,
        get_message_type=lambda: None,
        get_sender_id=lambda: "10001",
        get_sender_name=lambda: "主人",
        get_self_id=lambda: "bot",
        get_platform_name=lambda: "aiocqhttp",
        get_extra=lambda key, default=None: default,
    )


def _manual_handler(processor, messages, message_count: int):
    session_metadata = {"last_summarized_index": 0, "pending_summary": None}

    async def _get(_session_id, key, default=None):
        return session_metadata.get(key, default)

    async def _update(_session_id, key, value):
        session_metadata[key] = value

    manager = MagicMock()
    manager.store = MagicMock()
    manager.store.get_message_count = AsyncMock(return_value=message_count)
    manager.get_session_metadata = AsyncMock(side_effect=_get)
    manager.update_session_metadata = AsyncMock(side_effect=_update)
    manager.get_messages_range = AsyncMock(return_value=messages)
    engine = MagicMock()
    engine.add_memory = AsyncMock(return_value=1)
    handler = CommandHandler(
        context=MagicMock(),
        config_manager=ConfigManager(
            {
                "filtering_settings": {
                    "memory_scope_mode": "session",
                    "use_session_filtering": True,
                }
            }
        ),
        memory_engine=engine,
        conversation_manager=manager,
        index_validator=None,
        memory_processor=processor,
    )
    return handler, engine, session_metadata


@pytest.mark.asyncio
async def test_manual_summarize_assistant_only_advances_cursor_once():
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)
    handler, engine, session_metadata = _manual_handler(
        processor, _assistant_only_messages(), 3
    )
    event = _session_event()

    with patch(
        "astrbot_plugin_livingmemory.core.utils.get_persona_id",
        new=AsyncMock(return_value="persona-1"),
    ):
        first = [item async for item in handler.handle_summarize(event)]
        second = [item async for item in handler.handle_summarize(event)]

    assert any("未写入" in item for item in first)
    assert any("没有需要总结" in item for item in second)
    engine.add_memory.assert_not_awaited()
    llm.text_chat.assert_not_awaited()
    assert session_metadata["last_summarized_index"] == 3
    assert session_metadata["pending_summary"] is None


@pytest.mark.asyncio
async def test_manual_summarize_keeps_mixed_fact_in_session_scope():
    llm = _DummyLLMProvider(
        json.dumps(
            {
                "summary": "主人今天吃了牛肉面。我还觉得他好几周没联系我。",
                "topics": ["饮食"],
                "key_facts": [
                    "主人今天吃了牛肉面",
                    "主人已经好几周没有在任何平台联系过我",
                ],
                "sentiment": "neutral",
                "importance": 0.88,
            },
            ensure_ascii=False,
        )
    )
    messages = [
        _message(
            message_id=1,
            role="user",
            content="我今天吃了牛肉面",
            sender_id="10001",
            sender_name="主人",
            timestamp=time.time() - 3600,
        ),
        _message(
            message_id=2,
            role="assistant",
            content="你是不是所有平台都好几周没理我？",
            sender_id="bot",
            sender_name="Bot",
            timestamp=time.time() - 3500,
            bot=True,
        ),
    ]
    processor = MemoryProcessor(llm_provider=llm, context=None)
    handler, engine, _session_metadata = _manual_handler(processor, messages, 2)

    with patch(
        "astrbot_plugin_livingmemory.core.utils.get_persona_id",
        new=AsyncMock(return_value="persona-1"),
    ):
        _messages = [item async for item in handler.handle_summarize(_session_event())]

    engine.add_memory.assert_awaited_once()
    stored = engine.add_memory.await_args.kwargs
    assert stored["session_id"] == QQ_SESSION
    assert "牛肉面" in stored["content"]
    _assert_no_false_user_facts(stored["content"])
    assert all("任何平台" not in atom.content for atom in stored["atoms"])


@pytest.mark.asyncio
async def test_resummarize_assistant_only_source_does_not_replace_memory():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_record = AsyncMock(
        return_value={"id": 7, "metadata": {"persona_id": "persona-1", "session_id": QQ_SESSION}}
    )
    engine.get_memory_source = AsyncMock(
        return_value=[message.to_dict() for message in _assistant_only_messages()]
    )
    engine.replace_memory = AsyncMock(return_value=8)
    processor = MemoryProcessor(
        llm_provider=_DummyLLMProvider(_poisoned_summary()), context=None
    )
    request = MagicMock()
    request.get_json = AsyncMock(return_value={"memory_id": 7})

    with patch(
        "astrbot_plugin_livingmemory.core.page_api_modules.memory_handler.request",
        request,
    ):
        result = await handler.resummarize_memory(engine, processor)

    assert result["status"] == "ok"
    assert result["data"]["skipped"] is True
    assert result["data"]["new_memory_id"] == 7
    engine.replace_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_assistant_only_dialogue_does_not_create_memory():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_import_keys = AsyncMock(return_value=set())
    engine.add_memory = AsyncMock(return_value=1)
    processor = MemoryProcessor(
        llm_provider=_DummyLLMProvider(_poisoned_summary()), context=None
    )
    content = json.dumps(
        [
            {
                "role": "assistant",
                "content": "好久没见你了，你是不是好几周没联系我？",
                "sender_id": "bot",
            },
            {
                "role": "assistant",
                "content": "不管QQ还是微信，你是不是把我忘了？",
                "sender_id": "bot",
            },
        ],
        ensure_ascii=False,
    )
    request = MagicMock()
    request.get_json = AsyncMock(
        return_value={
            "format": "json",
            "content": content,
            "dry_run": False,
            "duplicate_strategy": "allow",
        }
    )

    with patch(
        "astrbot_plugin_livingmemory.core.page_api_modules.memory_handler_io.request",
        request,
    ):
        result = await handler.import_memories(engine, processor)

    assert result["status"] == "ok"
    assert result["data"]["imported_count"] == 0
    assert result["data"]["skipped_unattributed_count"] == 1
    engine.add_memory.assert_not_awaited()

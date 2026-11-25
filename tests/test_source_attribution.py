"""用户事实只从角色明确的用户消息里提取。

助手窗口直接跳过。混合窗口仍保存模型对用户原话的改写，
不用关键词或整句子串去删。模型凭空写出的内容不会被这层拦住，
对应测试把这个边界标出来。
"""

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
    select_user_evidence,
)

from tests.event_handler_support import _make_real_handler

QQ_SESSION = "aiocqhttp:FriendMessage:qq:10001"
GLOBAL_SCOPE = "livingmemory:global"
USER_FACT = "我这阵子心情不太好，李四跟我一个组"
USER_PARAPHRASE = "用户近期情绪低落，李四是同事"
ASSISTANT_SPECULATION = "你大概已经换了新工作，还搬去了另一个城市吧"
ASSISTANT_QUOTE = "用户亲口说过自己已经搬去了海港城"


class _DummyLLMProvider:
    def __init__(self, completion_text: str):
        self._completion_text = completion_text
        self.prompts: list[str] = []
        self.text_chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, prompt: str, system_prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(completion_text=self._completion_text)


def _completion(summary: str, facts: list[str], importance: float = 0.8) -> str:
    return json.dumps(
        {
            "summary": summary,
            "topics": ["近况"],
            "key_facts": facts,
            "sentiment": "neutral",
            "importance": importance,
        },
        ensure_ascii=False,
    )


def _message(
    *,
    message_id: int,
    role: str,
    content: str,
    sender_id: str,
    sender_name: str,
    timestamp: float | None = None,
    metadata: dict | None = None,
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
        timestamp=time.time() if timestamp is None else timestamp,
        metadata=metadata or {},
    )


def _user(content: str, message_id: int = 1) -> Message:
    return _message(
        message_id=message_id,
        role="user",
        content=content,
        sender_id="10001",
        sender_name="主人",
    )


def _assistant(content: str, message_id: int = 2) -> Message:
    return _message(
        message_id=message_id,
        role="assistant",
        content=content,
        sender_id="bot",
        sender_name="Bot",
        metadata={"is_bot_message": True},
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
            metadata={"is_bot_message": True},
        )
        for index, text in enumerate(texts)
    ]


def _poisoned_summary() -> str:
    return _completion(
        "主人已经好几周没有在任何平台联系我，我们的关系变淡了。",
        [
            "主人已经好几周没有在任何平台联系过我",
            "我们的关系变淡了",
            "主人心情很差",
        ],
        0.99,
    )


def _prompt(llm: _DummyLLMProvider) -> str:
    assert llm.prompts, "提取模型没有被调用"
    return llm.prompts[-1]


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
    graph = GraphExtractor().extract(1, content, metadata, atoms)
    assert atoms == []
    assert graph.entries == []
    assert graph.nodes == []


@pytest.mark.asyncio
async def test_empty_message_list_still_raises():
    processor = MemoryProcessor(
        llm_provider=_DummyLLMProvider(_poisoned_summary()), context=None
    )
    with pytest.raises(ValueError, match="消息列表不能为空"):
        await processor.process_conversation([])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        _message(
            message_id=1,
            role="user",
            content="[系统任务：主动对话]\n我们好像已经有一段时间没有说话了。",
            sender_id="10001",
            sender_name="主人",
        ),
        _message(
            message_id=1,
            role="user",
            content="在吗，今天过得怎么样",
            sender_id="10001",
            sender_name="主人",
            metadata={"is_proactive": True},
        ),
        _message(
            message_id=1,
            role="user",
            content="在吗，今天过得怎么样",
            sender_id="bot",
            sender_name="Bot",
            metadata={"is_bot_message": True},
        ),
        _message(
            message_id=1,
            role="bot",
            content="在吗，今天过得怎么样",
            sender_id="bot",
            sender_name="Bot",
        ),
    ],
)
async def test_proactive_or_bot_provenance_is_not_user_evidence(message):
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _content, metadata, _importance = await processor.process_conversation([message])

    llm.text_chat.assert_not_awaited()
    assert metadata["source_attribution"]["store_user_facts"] is False
    assert processor.classify_atoms_from_metadata(metadata) == []


@pytest.mark.asyncio
async def test_same_text_counts_only_with_user_role():
    llm = _DummyLLMProvider(_completion(USER_PARAPHRASE, [USER_PARAPHRASE]))
    processor = MemoryProcessor(llm_provider=llm, context=None)
    text = "我这阵子心情不太好"

    skipped, skipped_meta, _ = await processor.process_conversation(
        [_assistant(text, message_id=1)]
    )
    kept, kept_meta, _ = await processor.process_conversation([_user(text)])

    assert skipped == ""
    assert skipped_meta["source_attribution"]["assistant_only"] is True
    assert text in _prompt(llm)
    assert USER_PARAPHRASE in kept
    assert kept_meta["source_attribution"]["evidence_session_ids"] == [QQ_SESSION]


@pytest.mark.asyncio
async def test_ordinary_homework_sentence_is_not_treated_as_proactive_template():
    llm = _DummyLLMProvider(_completion("用户在做系统任务的作业", ["用户在做系统任务的作业"]))
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _importance = await processor.process_conversation(
        [_user("我在做系统任务的作业")]
    )

    assert "我在做系统任务的作业" in _prompt(llm)
    assert "系统任务的作业" in content
    assert metadata["source_attribution"]["store_user_facts"] is True


@pytest.mark.asyncio
async def test_non_verbatim_user_paraphrase_is_kept_and_assistant_text_is_not_input():
    llm = _DummyLLMProvider(_completion(USER_PARAPHRASE, [USER_PARAPHRASE], 0.91))
    processor = MemoryProcessor(llm_provider=llm, context=None)
    messages = [
        _user(USER_FACT),
        _assistant(ASSISTANT_SPECULATION),
    ]

    content, metadata, importance = await processor.process_conversation(messages)
    prompt = _prompt(llm)

    assert USER_FACT in prompt
    assert ASSISTANT_SPECULATION not in prompt
    assert "另一个城市" not in prompt
    assert USER_PARAPHRASE in content
    assert "情绪低落" in content
    assert "同事" in content
    assert USER_FACT not in content
    assert importance == 0.91
    assert metadata["source_attribution"]["evidence_session_ids"] == [QQ_SESSION]
    atoms = processor.classify_atoms_from_metadata(
        metadata, session_id=QQ_SESSION, persona_id="persona-1"
    )
    assert atoms
    assert all(atom.session_id == QQ_SESSION for atom in atoms)
    graph = GraphExtractor().extract(3, content, metadata, atoms)
    blob = "\n".join(entry.content for entry in graph.entries)
    assert "情绪低落" in blob or "同事" in blob


@pytest.mark.asyncio
async def test_model_hallucination_is_not_keyword_censored():
    """模型可以输出输入里没有的话。这层只保证助手原文没进入输入。"""
    invented = "用户已经搬去了另一个城市"
    llm = _DummyLLMProvider(_completion(invented, [invented], 0.77))
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, _metadata, importance = await processor.process_conversation(
        [_user("午饭我选了牛肉面"), _assistant(ASSISTANT_SPECULATION)]
    )

    assert "牛肉面" in _prompt(llm)
    assert "另一个城市" not in _prompt(llm)
    assert invented in content
    assert importance == 0.77


@pytest.mark.asyncio
async def test_assistant_quote_is_not_evidence_but_user_correction_is():
    llm = _DummyLLMProvider(
        _completion("用户澄清自己仍在原来的城市", ["用户澄清自己仍在原来的城市"])
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _importance = await processor.process_conversation(
        [
            _assistant(ASSISTANT_QUOTE, message_id=1),
            _user("你刚才猜我搬走了，其实我还在原来的城市", message_id=2),
        ]
    )
    prompt = _prompt(llm)

    assert "原来的城市" in prompt
    assert "亲口说过" not in prompt
    assert "海港城" not in prompt
    assert "原来的城市" in content
    assert metadata["source_attribution"]["user_message_count"] == 1
    assert metadata["source_attribution"]["assistant_message_count"] == 1


@pytest.mark.asyncio
async def test_channel_scope_stays_on_session_when_user_mentions_other_platforms():
    stated = "用户说自己所有平台都没怎么上，QQ 和微信都很少打开"
    llm = _DummyLLMProvider(_completion(stated, [stated]))
    processor = MemoryProcessor(llm_provider=llm, context=None)
    user_text = "我所有平台都没怎么上，QQ 和微信都很少打开"

    content, metadata, _importance = await processor.process_conversation(
        [_user(user_text), _assistant("那你是不是哪里都不在了？")]
    )

    assert user_text in _prompt(llm)
    assert "哪里都不在了" not in _prompt(llm)
    assert "所有平台" in content
    assert metadata["source_attribution"]["evidence_session_ids"] == [QQ_SESSION]
    assert GLOBAL_SCOPE not in metadata["source_attribution"]["evidence_session_ids"]
    atoms = processor.classify_atoms_from_metadata(metadata, session_id=QQ_SESSION)
    assert all(atom.session_id == QQ_SESSION for atom in atoms)


def test_system_and_media_placeholders_do_not_count_as_user_evidence():
    evidence, selected = select_user_evidence(
        [
            _message(
                message_id=1,
                role="system",
                content="你是助手",
                sender_id="system",
                sender_name="system",
            ),
            _user("[图片]"),
            _user("午饭我选了牛肉面", message_id=3),
        ]
    )
    assert evidence.user_count == 1
    assert len(selected) == 1
    assert selected[0].content == "午饭我选了牛肉面"


def test_graph_blocks_unattributed_result_and_keeps_historical_facts():
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
        ["早安呀。", "你是不是好几周都没来找我？", "不管QQ还是微信，你是不是把我忘了？"]
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
    handler._memory_reflection.memory_processor = MemoryProcessor(
        llm_provider=llm, context=None
    )
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)
    messages = await manager.get_messages_range(
        session_id=session_id, start_index=0, end_index=3
    )
    try:
        committed = await handler._memory_reflection._storage_task(
            session_id, messages, "persona-1", 0, 3, 0, memory_scope=session_id
        )
        assert committed is True
        llm.text_chat.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 3
        assert await manager.get_session_metadata(session_id, "pending_summary", None) is None

        again = await handler._memory_reflection._storage_task(
            session_id, messages, "persona-1", 0, 3, 0, memory_scope=session_id
        )
        assert again is False
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_reflection_empty_window_advances_cursor_without_loop(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    await manager.add_message(
        session_id=session_id,
        role="user",
        content="午饭我选了牛肉面",
        sender_id="10001",
        sender_name="主人",
        platform="aiocqhttp",
    )
    llm = _DummyLLMProvider(_poisoned_summary())
    handler._memory_reflection.memory_processor = MemoryProcessor(
        llm_provider=llm, context=None
    )
    handler.memory_engine.add_memory = AsyncMock(return_value=1)
    try:
        committed = await handler._memory_reflection._storage_task(
            session_id, [], "persona-1", 0, 1, 0, memory_scope=session_id
        )
        assert committed is True
        llm.text_chat.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 1
        again = await handler._memory_reflection._storage_task(
            session_id, [], "persona-1", 0, 1, 0, memory_scope=session_id
        )
        assert again is False
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_reflection_cursor_past_end_is_clamped_without_reprocessing(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    await manager.add_message(
        session_id=session_id,
        role="user",
        content="午饭我选了牛肉面",
        sender_id="10001",
        sender_name="主人",
        platform="aiocqhttp",
    )
    await manager.update_session_metadata(session_id, "last_summarized_index", 99)
    llm = _DummyLLMProvider(_poisoned_summary())
    handler._memory_reflection.memory_processor = MemoryProcessor(
        llm_provider=llm, context=None
    )
    try:
        plan = await handler._memory_reflection._build_next_reflection_plan(
            session_id, "persona-1", session_id, None
        )
        assert plan is None
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 1
        llm.text_chat.assert_not_awaited()
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_reflection_negative_cursor_restarts_at_zero_then_skips_assistant_only(
    tmp_path,
):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    base = time.time() - 86400
    for index, text in enumerate(["早安。", "在吗？"]):
        await manager.add_message(
            session_id=session_id,
            role="assistant",
            content=text,
            sender_id="bot",
            sender_name="Bot",
            platform="aiocqhttp",
            is_bot_message=True,
            timestamp=base + index,
        )
    await manager.update_session_metadata(session_id, "last_summarized_index", -4)
    llm = _DummyLLMProvider(_poisoned_summary())
    handler._memory_reflection.memory_processor = MemoryProcessor(
        llm_provider=llm, context=None
    )
    handler.memory_engine.add_memory = AsyncMock(return_value=1)
    try:
        plan = await handler._memory_reflection._build_next_reflection_plan(
            session_id, "persona-1", session_id, None
        )
        assert plan is not None
        assert plan.start_index == 0
        assert plan.end_index == 2
        committed = await handler._memory_reflection._storage_task(
            session_id,
            plan.history_messages,
            "persona-1",
            plan.start_index,
            plan.end_index,
            0,
            memory_scope=session_id,
        )
        assert committed is True
        llm.text_chat.assert_not_awaited()
        handler.memory_engine.add_memory.assert_not_awaited()
        assert await manager.get_session_metadata(session_id, "last_summarized_index", 0) == 2
        follow = await handler._memory_reflection._build_next_reflection_plan(
            session_id, "persona-1", session_id, None
        )
        assert follow is None
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_reflection_mixed_window_keeps_paraphrase_in_session_scope(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = QQ_SESSION
    base = time.time() - 86400
    await manager.add_message(
        session_id=session_id,
        role="user",
        content=USER_FACT,
        sender_id="10001",
        sender_name="主人",
        platform="aiocqhttp",
        timestamp=base,
    )
    await manager.add_message(
        session_id=session_id,
        role="assistant",
        content=ASSISTANT_SPECULATION,
        sender_id="bot",
        sender_name="Bot",
        platform="aiocqhttp",
        is_bot_message=True,
        timestamp=base + 60,
    )
    llm = _DummyLLMProvider(_completion(USER_PARAPHRASE, [USER_PARAPHRASE], 0.9))
    handler._memory_reflection.memory_processor = MemoryProcessor(
        llm_provider=llm, context=None
    )
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=11)
    messages = await manager.get_messages_range(
        session_id=session_id, start_index=0, end_index=2
    )
    try:
        committed = await handler._memory_reflection._storage_task(
            session_id, messages, "persona-1", 0, 2, 0, memory_scope=session_id
        )
        assert committed is True
        assert USER_FACT in _prompt(llm)
        assert "另一个城市" not in _prompt(llm)
        stored = handler.memory_engine.add_memory.await_args.kwargs
        assert stored["session_id"] == QQ_SESSION
        assert stored["session_id"] != GLOBAL_SCOPE
        assert USER_PARAPHRASE in stored["content"]
        assert stored["metadata"]["source_session_id"] == QQ_SESSION
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


def _manual_handler(processor, messages, message_count: int, cursor=0):
    session_metadata = {"last_summarized_index": cursor, "pending_summary": None}

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
async def test_manual_cursor_past_end_does_not_call_model():
    llm = _DummyLLMProvider(_poisoned_summary())
    handler, engine, session_metadata = _manual_handler(
        MemoryProcessor(llm_provider=llm, context=None),
        _assistant_only_messages(),
        3,
        cursor=99,
    )

    with patch(
        "astrbot_plugin_livingmemory.core.utils.get_persona_id",
        new=AsyncMock(return_value="persona-1"),
    ):
        messages = [item async for item in handler.handle_summarize(_session_event())]

    assert any("没有需要总结" in item for item in messages)
    assert session_metadata["last_summarized_index"] == 3
    engine.add_memory.assert_not_awaited()
    llm.text_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_negative_cursor_and_empty_fetch_do_not_write_facts():
    llm = _DummyLLMProvider(_poisoned_summary())
    processor = MemoryProcessor(llm_provider=llm, context=None)
    negative_handler, negative_engine, negative_meta = _manual_handler(
        processor, _assistant_only_messages(), 3, cursor=-8
    )
    empty_handler, empty_engine, empty_meta = _manual_handler(
        processor, [], 4, cursor=0
    )

    with patch(
        "astrbot_plugin_livingmemory.core.utils.get_persona_id",
        new=AsyncMock(return_value="persona-1"),
    ):
        negative_messages = [
            item async for item in negative_handler.handle_summarize(_session_event())
        ]
        empty_messages = [
            item async for item in empty_handler.handle_summarize(_session_event())
        ]

    assert any("未写入" in item for item in negative_messages)
    assert negative_meta["last_summarized_index"] == 3
    negative_engine.add_memory.assert_not_awaited()
    assert any("获取消息失败" in item for item in empty_messages)
    assert empty_meta["last_summarized_index"] == 0
    empty_engine.add_memory.assert_not_awaited()
    llm.text_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_summarize_keeps_paraphrase_in_session_scope():
    llm = _DummyLLMProvider(_completion(USER_PARAPHRASE, [USER_PARAPHRASE], 0.88))
    messages = [_user(USER_FACT), _assistant(ASSISTANT_SPECULATION)]
    processor = MemoryProcessor(llm_provider=llm, context=None)
    handler, engine, _session_metadata = _manual_handler(processor, messages, 2)

    with patch(
        "astrbot_plugin_livingmemory.core.utils.get_persona_id",
        new=AsyncMock(return_value="persona-1"),
    ):
        _messages = [item async for item in handler.handle_summarize(_session_event())]

    stored = engine.add_memory.await_args.kwargs
    assert stored["session_id"] == QQ_SESSION
    assert stored["session_id"] != GLOBAL_SCOPE
    assert USER_PARAPHRASE in stored["content"]
    assert "另一个城市" not in _prompt(llm)


@pytest.mark.asyncio
async def test_resummarize_assistant_only_source_does_not_replace_memory():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_record = AsyncMock(
        return_value={
            "id": 7,
            "metadata": {"persona_id": "persona-1", "session_id": QQ_SESSION},
        }
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
async def test_resummarize_short_source_returns_error_without_replacement():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_record = AsyncMock(return_value={"id": 7, "metadata": {}})
    engine.get_memory_source = AsyncMock(return_value=[_user("只有一句").to_dict()])
    engine.replace_memory = AsyncMock(return_value=8)
    request = MagicMock()
    request.get_json = AsyncMock(return_value={"memory_id": 7})

    with patch(
        "astrbot_plugin_livingmemory.core.page_api_modules.memory_handler.request",
        request,
    ):
        result = await handler.resummarize_memory(engine, MemoryProcessor(context=None))

    assert result["status"] == "error"
    engine.replace_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_resummarize_keeps_paraphrase_of_user_source():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_record = AsyncMock(
        return_value={"id": 7, "metadata": {"persona_id": "persona-1"}}
    )
    engine.get_memory_source = AsyncMock(
        return_value=[
            _user(USER_FACT).to_dict(),
            _assistant(ASSISTANT_SPECULATION).to_dict(),
        ]
    )
    engine.replace_memory = AsyncMock(return_value=9)
    llm = _DummyLLMProvider(_completion(USER_PARAPHRASE, [USER_PARAPHRASE]))
    request = MagicMock()
    request.get_json = AsyncMock(return_value={"memory_id": 7})

    with patch(
        "astrbot_plugin_livingmemory.core.page_api_modules.memory_handler.request",
        request,
    ):
        result = await handler.resummarize_memory(
            engine, MemoryProcessor(llm_provider=llm, context=None)
        )

    assert result["status"] == "ok"
    assert result["data"]["new_memory_id"] == 9
    replaced = engine.replace_memory.await_args.kwargs
    assert USER_PARAPHRASE in replaced["content"]
    assert "另一个城市" not in _prompt(llm)


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
            {"role": "assistant", "content": "好久没见你了。", "sender_id": "bot"},
            {
                "role": "assistant",
                "content": "你是不是把我忘了？",
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


@pytest.mark.asyncio
async def test_import_prewritten_fact_keeps_text_and_session():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_import_keys = AsyncMock(return_value=set())
    engine.add_memory = AsyncMock(return_value=4)
    llm = _DummyLLMProvider(_poisoned_summary())
    content = json.dumps(
        {
            "content": "用户这周没怎么上QQ，微信仍在聊",
            "session_id": QQ_SESSION,
            "importance": 0.6,
        },
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
        result = await handler.import_memories(
            engine, MemoryProcessor(llm_provider=llm, context=None)
        )

    assert result["data"]["imported_count"] == 1
    stored = engine.add_memory.await_args.kwargs
    assert stored["content"] == "用户这周没怎么上QQ，微信仍在聊"
    assert stored["session_id"] == QQ_SESSION
    llm.text_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_invalid_entry_does_not_create_memory():
    handler = MemoryHandler(PageApiUtils())
    engine = MagicMock()
    engine.get_memory_import_keys = AsyncMock(return_value=set())
    engine.add_memory = AsyncMock(return_value=1)
    request = MagicMock()
    request.get_json = AsyncMock(
        return_value={
            "format": "json",
            "content": json.dumps([{"note": "没有正文也没有对话"}]),
            "dry_run": False,
            "duplicate_strategy": "allow",
        }
    )

    with patch(
        "astrbot_plugin_livingmemory.core.page_api_modules.memory_handler_io.request",
        request,
    ):
        result = await handler.import_memories(engine, MemoryProcessor(context=None))

    assert result["data"]["imported_count"] == 0
    assert result["data"]["failed_count"] == 1
    engine.add_memory.assert_not_awaited()

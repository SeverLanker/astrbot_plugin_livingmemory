#!/usr/bin/env python3
"""把来源归因修复补丁应用到已安装的 LivingMemory 2.8.4。

只接受与本次核对过的 2.8.4 文件指纹一致的目录。补丁不会升级到 2.15.0，
也不会改记忆作用域。应用前会把将被覆盖的文件复制到备份目录。

用法:
  python3 apply_284_source_attribution.py --root /path/to/astrbot_plugin_livingmemory
  python3 apply_284_source_attribution.py --root /path/to/plugin --check
  python3 apply_284_source_attribution.py --rollback /path/to/backup-dir

回滚:
  成功应用后脚本会打印备份目录。确认需要撤回时执行 --rollback，
  它会按清单恢复被覆盖的文件，并删除本次新增的文件。
  回滚不会触碰记忆数据库、向量或已经人工修正过的历史记忆。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PATCH_NAME = "livingmemory-2.8.4-source-attribution.patch"

# 部署目录在补丁前必须与这些指纹一致。memory_scope.py 只做哨兵，补丁不修改它。
EXPECTED_BEFORE = {
    "core/command_handler.py": "93cf841cea9652b1ffb31e3051e1612a1006fa405b9e9df3a8182b4cd8c09c66",
    "core/event_handler_modules/memory_reflection.py": "1c32fcc54d00fc111184c6ffb89951e77affadc9ed1de49f2b896ccd2ea88a17",
    "core/i18n/en.json": "dfb4b58cabb63b7c7b7d794ce306b803f9d6fd6a82202b5fbf0f478ed6bced5c",
    "core/i18n/ja.json": "fb5c5a7bde501cf133e3b775577bfd4efc71d366494403bad305a710cc33bafc",
    "core/i18n/ru.json": "c49b1ac22a604d198a865c5b6b32537a12e931c351cb03ae9d90d71594c3e474",
    "core/i18n/zh.json": "89829f3e13226a119f17509128dd64116a5537a811190f1ab944deec58ae5de1",
    "core/memory_scope.py": "35720e27f868c1fffaeba5e56b87d6eb29154193246653f9a84d025e812eaece",
    "core/page_api_modules/memory_handler.py": "bfbcff6dfa7fc1645ca943b3125ef4bafd6eb3b16e82ca97aabd5a4818e36c5d",
    "core/page_api_modules/memory_handler_io.py": "e1e7aeec0aa7f6939889cc12d82c407eaeee74139b5dd3babe1f981fd43931f6",
    "core/processors/graph_extractor.py": "27c098ebb5ab738ace74aaf190c7ea45a752bbb49b7c34c6f6b0b223b5cf684a",
    "core/processors/memory_processor.py": "3d720f029bd20c5c552fce42e1e177bb60733ea4f51b77d1f22939600fa27c11",
    "core/processors/memory_processor_build.py": "52dcfe07c82296ddbd56ff81212743b8293dada0982f9a0e52fddae1e099aeb6",
    "tests/conftest.py": "77388ce77bc9c2080b8dd7eb834d489b1f8087a4b3f8d940b57d2f5bc230de6d",
}
EXPECTED_AFTER = {
    "core/command_handler.py": "6c70e46b4b181c7fdf098ee7f4517d695c9bd6fdaeb942d55176bf1b4ea32993",
    "core/event_handler_modules/memory_reflection.py": "760655cfd3acd39b6bc9bca209f1ace1bf572672c1dc1ea93a75c594749adf68",
    "core/i18n/en.json": "40ac97ff0ddc8361d87a1427e681450d8ee0d95de0845a6ca836bf24775f7156",
    "core/i18n/ja.json": "eaa32f88e831cc935216a2fb0d316694bff4c181436d7fbbe66de7692f352f10",
    "core/i18n/ru.json": "2a62f840d1e5e8b060d198cd31d4c312f5da7f487b912b5e2ff13d2e1ef624ff",
    "core/i18n/zh.json": "b185f80ca8ea547dd335e59f89c17805519b39cf2ce17ec716c04b2e6bdc6704",
    "core/memory_scope.py": "35720e27f868c1fffaeba5e56b87d6eb29154193246653f9a84d025e812eaece",
    "core/page_api_modules/memory_handler.py": "794790070e4b49e7a322456e99048f8bd1de680e25f1ba1a87d401478293e998",
    "core/page_api_modules/memory_handler_io.py": "930046c14dedf0baa2a46c662be24c520a544ac95e16a4d5f17b0807393ef8a5",
    "core/processors/graph_extractor.py": "2bcce3be73f68d6c5a804265f0a87913ffaadebe3cef7de5d1486b93dc967492",
    "core/processors/memory_processor.py": "8f60451b5ef59c711dc1765351c9749c1c866e6b869b8833fd79a093f8c90ad8",
    "core/processors/memory_processor_build.py": "e497a0dd9a2b1453f53ee4ee237f4eb4a85bad8e9062b48f05c49e79f2943327",
    "core/processors/source_attribution.py": "d816042bbbdabae313b2d49e4c1c5f62d9fd5f60ef3cb0510e581e73a1afbc83",
    "tests/conftest.py": "2b4dbfe5c86a6c8a6e126784dd06b509e0bdb20cf7dcfa3df71f58c0e65b7001",
    "tests/test_source_attribution.py": "fa4ec4d0ebbe5884ca47d85b4ccc762f6cbdccabbf9f8f978715856feed60a9a",
}
CREATED_FILES = (
    "core/processors/source_attribution.py",
    "tests/test_source_attribution.py",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def mismatches(root: Path, expected: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for relative, digest in expected.items():
        actual = sha256(root / relative)
        if actual != digest:
            lines.append(f"{relative}\n  期望 {digest}\n  实际 {actual or '文件不存在'}")
    return lines


def already_applied(root: Path) -> bool:
    return not mismatches(root, EXPECTED_AFTER)


def backup(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f".livingmemory-284-attribution-backup-{stamp}"
    destination.mkdir(parents=True)
    replaced: list[str] = []
    for relative in EXPECTED_BEFORE:
        source = root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        replaced.append(relative)
    manifest = {
        "created_utc": stamp,
        "root": str(root),
        "replaced": replaced,
        "created": list(CREATED_FILES),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def apply_patch(root: Path, patch_path: Path) -> None:
    for relative in CREATED_FILES:
        if (root / relative).exists():
            raise SystemExit(f"拒绝覆盖已存在的新文件: {relative}")
    command = [
        "patch",
        "--forward",
        "--batch",
        "-p1",
        "--dry-run",
        "-i",
        str(patch_path),
    ]
    dry_run = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if dry_run.returncode != 0:
        raise SystemExit(
            "补丁预检失败，目录未修改。\n"
            + dry_run.stdout
            + dry_run.stderr
        )
    command.remove("--dry-run")
    applied = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if applied.returncode != 0:
        raise SystemExit(
            "补丁写入失败。请立刻用备份目录回滚。\n"
            + applied.stdout
            + applied.stderr
        )


def rollback(backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"备份清单不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["root"])
    if not root.is_dir():
        raise SystemExit(f"原插件目录不存在: {root}")
    for relative in manifest.get("created", []):
        target = root / relative
        if target.is_file():
            target.unlink()
    for relative in manifest.get("replaced", []):
        source = backup_dir / relative
        target = root / relative
        if not source.is_file():
            raise SystemExit(f"备份缺少文件: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"已从 {backup_dir} 回滚到 {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="应用 LivingMemory 2.8.4 来源归因补丁")
    parser.add_argument("--root", type=Path, help="已安装的 astrbot_plugin_livingmemory 目录")
    parser.add_argument("--check", action="store_true", help="只检查指纹，不写文件")
    parser.add_argument("--rollback", type=Path, help="用指定备份目录回滚")
    args = parser.parse_args()

    if args.rollback:
        rollback(args.rollback)
        return
    if args.root is None:
        parser.error("需要 --root 或 --rollback")

    root = args.root.resolve()
    patch_path = Path(__file__).resolve().parent / PATCH_NAME
    if not patch_path.is_file():
        raise SystemExit(f"找不到补丁文件: {patch_path}")
    if not (root / "metadata.yaml").is_file():
        raise SystemExit(f"不是插件目录: {root}")

    if already_applied(root):
        print("指纹已经是补丁后的状态，无需再次应用。")
        return

    problems = mismatches(root, EXPECTED_BEFORE)
    for relative in CREATED_FILES:
        if (root / relative).exists():
            problems.append(f"{relative}\n  期望 文件不存在\n  实际 已存在")
    if problems:
        raise SystemExit(
            "前置指纹不匹配，已停止，没有修改文件。\n" + "\n".join(problems)
        )
    if args.check:
        print("前置指纹匹配 2.8.4，可以应用补丁。")
        return

    backup_dir = backup(root)
    try:
        apply_patch(root, patch_path)
        problems = mismatches(root, EXPECTED_AFTER)
        if problems:
            raise RuntimeError("补丁后指纹不匹配:\n" + "\n".join(problems))
    except Exception as exc:
        print(f"应用失败: {exc}", file=sys.stderr)
        print(f"正在从 {backup_dir} 回滚。", file=sys.stderr)
        rollback(backup_dir)
        raise SystemExit(1) from exc
    print("补丁已应用，后置指纹匹配。")
    print(f"备份目录: {backup_dir}")
    print(f"回滚命令: python3 {Path(__file__).resolve()} --rollback {backup_dir}")
    print("回滚只恢复这次覆盖的源码，不会改记忆数据库。重启插件前请先确认。")


if __name__ == "__main__":
    main()

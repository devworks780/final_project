"""Распаковать исходный CSV из ZIP-архива для следующих шагов DVC.

CSV значительно больше архива, поэтому в DVC и S3 хранится ZIP, а распакованный
файл объявлен локальным промежуточным результатом с ``cache: false``.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from logging_utils import get_step_logger, run_logged_step

LOGGER = get_step_logger("unpack")


def parse_args() -> argparse.Namespace:
    """Прочитать пути, оставив значения по умолчанию для запуска через DVC."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("dvc_data/data/origin/train_ver2.csv.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dvc_data/data/origin/train_ver2.csv"),
    )
    parser.add_argument("--member", default="train_ver2.csv")
    return parser.parse_args()


def main() -> None:
    """Проверить архив и атомарно извлечь из него ожидаемый CSV-файл."""
    args = parse_args()
    LOGGER.info("Archive: %s; member: %s", args.archive, args.member)
    if not args.archive.is_file():
        raise FileNotFoundError(f"ZIP archive not found: {args.archive}")
    if Path(args.member).name != args.member:
        # Запрещаем пути вида ../../file: из архива разрешено извлечь только файл
        # с простым именем, а место назначения задаётся отдельным аргументом.
        raise ValueError("Archive member must be a plain filename")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(args.archive) as archive:
            members = {item.filename: item for item in archive.infolist()}
            if args.member not in members:
                available = ", ".join(sorted(members))
                raise FileNotFoundError(
                    f"{args.member} is absent from {args.archive}; found: {available}"
                )
            member = members[args.member]
            if member.is_dir():
                raise ValueError(f"Archive member is a directory: {args.member}")

            # Сначала пишем рядом с итоговым CSV во временный файл. Если процесс
            # прервётся, частичный CSV не будет принят следующими шагами пайплайна.
            with tempfile.NamedTemporaryFile(
                dir=args.output.parent,
                prefix=f".{args.output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with archive.open(member) as source:
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)

        # replace() на одном файловом разделе атомарно публикует готовый файл.
        temporary_path.replace(args.output)
    except (BadZipFile, OSError):
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise

    # Проверка размера ловит неполную распаковку и повреждённый результат даже
    # в тех случаях, когда чтение архива завершилось без явного исключения.
    extracted_size = args.output.stat().st_size
    if extracted_size != member.file_size:
        args.output.unlink(missing_ok=True)
        raise ValueError(
            f"Extracted size mismatch: expected {member.file_size}, got {extracted_size}"
        )
    LOGGER.info("Extracted %s bytes to %s", extracted_size, args.output)


if __name__ == "__main__":
    run_logged_step("unpack", main)

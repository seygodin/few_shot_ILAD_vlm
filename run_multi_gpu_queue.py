#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# 필요하면 이 딕셔너리만 수정해서 바로 실행 가능
# GPU마다 리스트 내부 명령은 "순차 실행", GPU 간에는 "동시 실행"
DEFAULT_GPU_COMMANDS = {
    0: [
        {
            "command": "python train_hybrid.py --data_name screw_bag --mask True --text True "
            "--double_encoder True --few_shot True --shot 5 --hybrid_model winclip "
            "--hybrid_weight 0.3 --hybrid_batch_size 16 --epoch 100 --log True "
            "--lr 5e-4 --detection True --preprocess True",
            "repeat": 1,
        },
    ],
    1: [],
    2: [],
    3: [],
}


def _expand_command_entries(gpu_id: int, entries: list) -> list[str]:
    expanded_commands: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            expanded_commands.append(entry)
            continue

        if isinstance(entry, dict):
            command = entry.get("command", entry.get("cmd"))
            repeat = entry.get("repeat", entry.get("n", 1))

            if not isinstance(command, str) or command.strip() == "":
                raise ValueError(f"GPU {gpu_id} command entry has invalid command: {entry}")
            if not isinstance(repeat, int) or repeat < 1:
                raise ValueError(f"GPU {gpu_id} command entry has invalid repeat: {entry}")

            expanded_commands.extend([command] * repeat)
            continue

        raise ValueError(
            f"GPU {gpu_id} command must be str or dict(command/repeat). Got: {entry}"
        )

    return expanded_commands


def _load_gpu_commands(config_path: str | None) -> dict[int, list[str]]:
    if config_path is None:
        source = DEFAULT_GPU_COMMANDS
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            source = json.load(f)

    commands: dict[int, list[str]] = {}
    for key, value in source.items():
        gpu_id = int(key)
        if not isinstance(value, list):
            raise ValueError(
                f"GPU {gpu_id} value must be a list of command entries (str or dict)."
            )
        commands[gpu_id] = _expand_command_entries(gpu_id, value)
    return commands


def _run_one_command(
    *,
    gpu_id: int,
    cmd_index: int,
    total_cmds: int,
    command: str,
    run_dir: Path,
    dry_run: bool,
) -> dict:
    gpu_dir = run_dir / f"gpu_{gpu_id}"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    log_path = gpu_dir / f"cmd_{cmd_index:03d}.log"

    started_at = datetime.now()
    start_time = time.time()
    return_code = 0

    if dry_run:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[DRY RUN] gpu={gpu_id} ({cmd_index}/{total_cmds})\n")
            f.write(f"{command}\n")
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[START] {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[GPU] {gpu_id}\n")
            f.write(f"[INDEX] {cmd_index}/{total_cmds}\n")
            f.write(f"[COMMAND] {command}\n\n")
            f.flush()

            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(Path.cwd()),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                executable="/bin/bash",
            )
            return_code = process.wait()

            finished_at = datetime.now()
            f.write("\n")
            f.write(f"[END] {finished_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[RETURN CODE] {return_code}\n")

    elapsed = time.time() - start_time
    return {
        "gpu_id": gpu_id,
        "cmd_index": cmd_index,
        "total_cmds": total_cmds,
        "command": command,
        "return_code": return_code,
        "elapsed_sec": round(elapsed, 3),
        "log_path": str(log_path),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": int(dry_run),
    }


def _gpu_worker(
    *,
    gpu_id: int,
    commands: list[str],
    run_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> list[dict]:
    results: list[dict] = []
    total_cmds = len(commands)

    if total_cmds == 0:
        return results

    print(f"[GPU {gpu_id}] queue start: {total_cmds} commands")
    for idx, command in enumerate(commands, start=1):
        print(f"[GPU {gpu_id}] run {idx}/{total_cmds}")
        result = _run_one_command(
            gpu_id=gpu_id,
            cmd_index=idx,
            total_cmds=total_cmds,
            command=command,
            run_dir=run_dir,
            dry_run=dry_run,
        )
        results.append(result)

        rc = result["return_code"]
        print(f"[GPU {gpu_id}] done {idx}/{total_cmds} (rc={rc})")
        if rc != 0 and not continue_on_error:
            print(f"[GPU {gpu_id}] stop queue due to failure")
            break

    print(f"[GPU {gpu_id}] queue finished")
    return results


def _write_summary(run_dir: Path, all_results: list[dict]) -> None:
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "gpu_id",
        "cmd_index",
        "total_cmds",
        "return_code",
        "elapsed_sec",
        "started_at",
        "dry_run",
        "log_path",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "GPU별 명령 큐를 실행합니다. 같은 GPU는 직렬, 다른 GPU는 병렬 실행합니다."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default='run_multi_gpu_command_config.json',
        help=(
            "JSON 파일 경로. 형식: "
            "{\"0\": [\"cmd1\", {\"command\": \"cmd2\", \"repeat\": 3}], \"1\": []}. "
            "없으면 스크립트 내부 DEFAULT_GPU_COMMANDS 사용."
        ),
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="./logs/multi_gpu_queue",
        help="실행 로그 루트 디렉토리",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실행하지 않고 로그/요약만 생성",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="명령 실패(return_code!=0)해도 같은 GPU 큐에서 다음 명령 계속 실행",
    )
    args = parser.parse_args()

    gpu_commands = _load_gpu_commands(args.config)
    active_gpu_commands = {k: v for k, v in gpu_commands.items() if len(v) > 0}
    if len(active_gpu_commands) == 0:
        print("실행할 명령이 없습니다. DEFAULT_GPU_COMMANDS 또는 --config를 확인하세요.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(Path.cwd()),
        "dry_run": int(args.dry_run),
        "continue_on_error": int(args.continue_on_error),
        "gpu_commands": gpu_commands,
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    all_results: list[dict] = []
    max_workers = len(active_gpu_commands)
    print(f"[RUN] active_gpus={sorted(active_gpu_commands.keys())}, workers={max_workers}")
    print(f"[RUN] log_dir={run_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _gpu_worker,
                gpu_id=gpu_id,
                commands=commands,
                run_dir=run_dir,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            ): gpu_id
            for gpu_id, commands in active_gpu_commands.items()
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                gpu_results = future.result()
                all_results.extend(gpu_results)
            except Exception as err:
                print(f"[GPU {gpu_id}] worker error: {err}")

    all_results.sort(key=lambda x: (x["gpu_id"], x["cmd_index"]))
    _write_summary(run_dir, all_results)

    failures = [r for r in all_results if r["return_code"] != 0]
    print(f"[DONE] total={len(all_results)}, failed={len(failures)}")
    print(f"[DONE] summary={run_dir / 'summary.csv'}")
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import csv
import json
import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from pdb import set_trace

# 새 형식 예시:
# {
#   "gpus": [0, 1, 2, 3],
#   "tasks": [
#     {"name": "breakfast_winclip", "command": "python train_hybrid.py ...", "repeat": 10},
#     {"command": "python train_hybrid.py ...", "repeat": 5}
#   ]
# }
#
# 레거시 형식(기존 run_multi_gpu_command_config.json)도 지원:
# {"0": [...], "1": [...], "2": [...], "3": [...]}
DEFAULT_CONFIG = {
    "gpus": [0, 1, 2, 3],
    "tasks": [
        {
            "name": "sample_screw_bag_winclip",
            "command": "python train_hybrid.py --data_name screw_bag --mask True --text True "
            "--double_encoder True --few_shot True --shot 5 --hybrid_model winclip "
            "--hybrid_weight 0.3 --hybrid_batch_size 16 --epoch 100 --log True "
            "--lr 5e-4 --detection True --preprocess True",
            "repeat": 1,
        }
    ],
}


def _parse_gpu_ids(gpu_str: str) -> list[int]:
    gpu_ids = []
    for token in gpu_str.split(","):
        token = token.strip()
        if token == "":
            continue
        gpu_ids.append(int(token))
    if len(gpu_ids) == 0:
        raise ValueError("--gpus must include at least one gpu id.")
    return gpu_ids


def _parse_task_entry(entry, source_label: str) -> tuple[str, int, str]:
    if isinstance(entry, str):
        return entry, 1, source_label

    if not isinstance(entry, dict):
        raise ValueError(f"Invalid task entry in {source_label}: {entry}")

    command = entry.get("command", entry.get("cmd"))
    repeat = entry.get("repeat", entry.get("n", 1))
    name = entry.get("name", source_label)

    if not isinstance(command, str) or command.strip() == "":
        raise ValueError(f"Invalid command in {source_label}: {entry}")
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"Invalid repeat in {source_label}: {entry}")
    if not isinstance(name, str) or name.strip() == "":
        name = source_label

    return command, repeat, name


def _expand_tasks(task_entries: list, source_prefix: str) -> list[dict]:
    tasks: list[dict] = []
    for base_idx, entry in enumerate(task_entries, start=1):
        source_label = f"{source_prefix}_{base_idx}"
        command, repeat, name = _parse_task_entry(entry, source_label)
        for repeat_idx in range(1, repeat + 1):
            tasks.append(
                {
                    "task_id": len(tasks) + 1,
                    "base_index": base_idx,
                    "task_name": name,
                    "repeat_index": repeat_idx,
                    "repeat_total": repeat,
                    "command": command,
                }
            )
    return tasks


def _load_config(config_path: str | None) -> tuple[list[int], list[dict], dict]:
    if config_path is None:
        raw = DEFAULT_CONFIG
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    # 새 형식
    if "tasks" in raw:
        from pdb import set_trace
        #set_trace()
        gpus = raw.get("gpus", [0, 1, 2, 3])
        if not isinstance(gpus, list):
            raise ValueError("'gpus' must be a list of integers.")
        gpu_ids = [int(g) for g in gpus]

        task_entries = raw["tasks"]
        if not isinstance(task_entries, list):
            raise ValueError("'tasks' must be a list.")
        tasks = _expand_tasks(task_entries, "task")
        return gpu_ids, tasks, raw

    # 레거시 형식 {"0":[...], "1":[...]} -> 전체 큐로 평탄화
    gpu_ids = sorted(int(k) for k in raw.keys())
    merged_entries = []
    for gpu_key in sorted(raw.keys(), key=lambda x: int(x)):
        value = raw[gpu_key]
        if not isinstance(value, list):
            raise ValueError(f"GPU {gpu_key} value must be a list.")
        merged_entries.extend(value)
    tasks = _expand_tasks(merged_entries, "legacy")
    return gpu_ids, tasks, raw


def _run_one_task(
    *,
    gpu_id: int,
    task: dict,
    total_tasks: int,
    run_dir: Path,
    dry_run: bool,
) -> dict:
    gpu_dir = run_dir / f"gpu_{gpu_id}"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    log_path = gpu_dir / f"task_{task['task_id']:04d}.log"

    started_at = datetime.now()
    start_time = time.time()
    return_code = 0
    command = task["command"]

    if dry_run:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[DRY RUN] gpu={gpu_id}\n")
            f.write(f"[TASK] {task['task_id']}/{total_tasks}\n")
            f.write(f"[NAME] {task['task_name']}\n")
            f.write(f"[REPEAT] {task['repeat_index']}/{task['repeat_total']}\n")
            f.write(f"[COMMAND] {command}\n")
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"[START] {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[GPU] {gpu_id}\n")
            f.write(f"[TASK] {task['task_id']}/{total_tasks}\n")
            f.write(f"[NAME] {task['task_name']}\n")
            f.write(f"[REPEAT] {task['repeat_index']}/{task['repeat_total']}\n")
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
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "base_index": task["base_index"],
        "repeat_index": task["repeat_index"],
        "repeat_total": task["repeat_total"],
        "gpu_id": gpu_id,
        "return_code": return_code,
        "elapsed_sec": round(elapsed, 3),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": int(dry_run),
        "log_path": str(log_path),
        "command": command,
    }


def _gpu_worker(
    *,
    gpu_id: int,
    task_queue: queue.Queue,
    total_tasks: int,
    run_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
    stop_event: Event,
    result_list: list[dict],
    result_lock: Lock,
) -> None:
    print(f"[GPU {gpu_id}] worker start")
    while True:
        if stop_event.is_set() and not continue_on_error:
            break

        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[GPU {gpu_id}] start task {task['task_id']}/{total_tasks}")
        result = _run_one_task(
            gpu_id=gpu_id,
            task=task,
            total_tasks=total_tasks,
            run_dir=run_dir,
            dry_run=dry_run,
        )
        with result_lock:
            result_list.append(result)

        rc = result["return_code"]
        print(f"[GPU {gpu_id}] done task {task['task_id']}/{total_tasks} (rc={rc})")
        if rc != 0 and not continue_on_error:
            stop_event.set()

    print(f"[GPU {gpu_id}] worker finish")


def _write_summary(run_dir: Path, all_results: list[dict]) -> None:
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "task_id",
        "task_name",
        "base_index",
        "repeat_index",
        "repeat_total",
        "gpu_id",
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
            "전체 작업 큐에서 유휴 GPU가 다음 작업을 가져가 실행합니다. "
            "GPU 간 병렬, GPU당 동시 1개 작업."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="run_multi_gpu_work_config.json",
        help=(
            "JSON 경로. 새 형식({gpus, tasks})과 레거시 형식({\"0\": [...], ...}) 모두 지원."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU 목록 강제 지정. 예: --gpus 0,1,2,3",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="./logs/multi_gpu_work_queue",
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
        help="명령 실패(return_code!=0) 후에도 큐 실행 계속",
    )
    args = parser.parse_args()

    gpu_ids, tasks, raw_config = _load_config(args.config)
    if args.gpus is not None:
        gpu_ids = _parse_gpu_ids(args.gpus)

    if len(gpu_ids) == 0:
        print("사용할 GPU가 없습니다.")
        return 1
    if len(tasks) == 0:
        print("실행할 작업이 없습니다.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(Path.cwd()),
        "dry_run": int(args.dry_run),
        "continue_on_error": int(args.continue_on_error),
        "gpu_ids": gpu_ids,
        "task_count": len(tasks),
        "config_path": args.config,
        "raw_config": raw_config,
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    task_queue: queue.Queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    all_results: list[dict] = []
    result_lock = Lock()
    stop_event = Event()

    #set_trace()

    print(f"[RUN] gpus={gpu_ids}, total_tasks={len(tasks)}")
    print(f"[RUN] log_dir={run_dir}")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(
                _gpu_worker,
                gpu_id=gpu_id,
                task_queue=task_queue,
                total_tasks=len(tasks),
                run_dir=run_dir,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
                stop_event=stop_event,
                result_list=all_results,
                result_lock=result_lock,
            )
            for gpu_id in gpu_ids
        ]
        for future in futures:
            future.result()

    all_results.sort(key=lambda x: x["task_id"])
    _write_summary(run_dir, all_results)

    failures = [r for r in all_results if r["return_code"] != 0]
    remaining_tasks = len(tasks) - len(all_results)
    print(f"[DONE] executed={len(all_results)}, failed={len(failures)}, remaining={remaining_tasks}")
    print(f"[DONE] summary={run_dir / 'summary.csv'}")

    if failures:
        return 2
    if remaining_tasks > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


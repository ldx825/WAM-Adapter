import bisect
import json
from pathlib import Path
from typing import Any, Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class TaskEpisodeWindowSampler(Sampler[int]):
    """Uniformly sample task -> episode -> legal temporal window.

    This is the task-balanced sampler used by the RoboTwin five-task runs.  Unlike flat dataset shuffling, each task is
    selected equally often even when its episodes are much longer.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        manifest_path: str | Path,
        seed: int,
        batch_size: int,
        num_processes: int,
        expected_world_size: int,
        samples_per_epoch: int,
        expected_task_count: int,
        episodes_per_task: int,
    ):
        self.dataset = dataset
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.expected_world_size = int(expected_world_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.expected_task_count = int(expected_task_count)
        self.episodes_per_task = int(episodes_per_task)
        self.epoch = 0
        self.resume_batch_offset = 0

        if self.num_processes != self.expected_world_size:
            raise ValueError(
                f"TaskEpisodeWindowSampler requires world_size={self.expected_world_size}, "
                f"got {self.num_processes}."
            )
        global_microbatch = self.batch_size * self.num_processes
        if self.samples_per_epoch <= 0 or self.samples_per_epoch % global_microbatch != 0:
            raise ValueError(
                "samples_per_epoch must be positive and divisible by the global microbatch: "
                f"samples={self.samples_per_epoch}, global_microbatch={global_microbatch}."
            )

        manifest_file = Path(self.manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Missing task sampler manifest: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.tasks = list(manifest.get("selected_tasks") or manifest.get("tasks") or [])
        if len(self.tasks) != self.expected_task_count or len(set(self.tasks)) != len(self.tasks):
            raise ValueError(f"Expected {self.expected_task_count} unique tasks, got {self.tasks}")
        if self.samples_per_epoch % len(self.tasks) != 0:
            raise ValueError("samples_per_epoch must be divisible by the task count")

        task_to_episodes: dict[str, list[int]] = {task: [] for task in self.tasks}
        for row in manifest.get("episodes", []):
            if row.get("split", "train") != "train":
                continue
            task = row.get("canonical_task_id", row.get("task_name"))
            episode = row.get("source_episode_index", row.get("episode_index"))
            if task not in task_to_episodes or episode is None:
                raise ValueError(f"Invalid episode row in task sampler manifest: {row}")
            task_to_episodes[task].append(int(episode))
        counts = {task: len(episodes) for task, episodes in task_to_episodes.items()}
        if any(count != self.episodes_per_task for count in counts.values()):
            raise ValueError(f"Expected {self.episodes_per_task} episodes per task: {counts}")
        all_episodes = [episode for episodes in task_to_episodes.values() for episode in episodes]
        if len(set(all_episodes)) != self.expected_task_count * self.episodes_per_task:
            raise ValueError("Duplicate source episode index in task sampler manifest")

        if not hasattr(dataset, "get_source_episode_sample_ranges") or not hasattr(dataset, "num_frames"):
            raise TypeError("TaskEpisodeWindowSampler requires RobotVideoDataset range helpers")
        source_ranges = dataset.get_source_episode_sample_ranges()
        if set(source_ranges) != set(all_episodes):
            missing = sorted(set(all_episodes).difference(source_ranges))
            extra = sorted(set(source_ranges).difference(all_episodes))
            raise ValueError(
                "Dataset episodes do not match task sampler manifest: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        self.window_size = int(dataset.num_frames)
        self.task_to_episodes = {
            task: tuple(sorted(episodes)) for task, episodes in task_to_episodes.items()
        }
        episode_to_task = {
            episode: task for task, episodes in self.task_to_episodes.items() for episode in episodes
        }
        self.episode_ranges: dict[int, tuple[int, int]] = {}
        self._range_starts: list[int] = []
        self._range_records: list[tuple[int, int, int, str]] = []
        for episode, (sample_start, sample_end) in sorted(source_ranges.items(), key=lambda x: x[1][0]):
            legal_stop = int(sample_end) - self.window_size + 1
            if legal_stop <= int(sample_start):
                raise ValueError(f"Episode {episode} has no legal {self.window_size}-frame window")
            record = (int(sample_start), legal_stop, int(episode), episode_to_task[int(episode)])
            self.episode_ranges[int(episode)] = record[:2]
            self._range_starts.append(record[0])
            self._range_records.append(record)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.set_epoch(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        if batch_in_epoch < 0:
            raise ValueError("resume batch offset must be non-negative")
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _sequence(self) -> list[int]:
        task_generator = torch.Generator(device="cpu")
        episode_generator = torch.Generator(device="cpu")
        window_generator = torch.Generator(device="cpu")
        task_generator.manual_seed(self.seed + self.epoch)
        episode_generator.manual_seed(self.seed + self.epoch + 1_000_003)
        window_generator.manual_seed(self.seed + self.epoch + 2_000_033)

        task_sequence: list[str] = []
        for _ in range(self.samples_per_epoch // len(self.tasks)):
            permutation = torch.randperm(len(self.tasks), generator=task_generator).tolist()
            task_sequence.extend(self.tasks[index] for index in permutation)

        indices: list[int] = []
        for task in task_sequence:
            episodes = self.task_to_episodes[task]
            episode = episodes[int(torch.randint(len(episodes), (1,), generator=episode_generator))]
            start, legal_stop = self.episode_ranges[episode]
            indices.append(int(torch.randint(start, legal_stop, (1,), generator=window_generator)))
        return indices

    def __iter__(self) -> Iterator[int]:
        indices = self._sequence()
        offset = self.resume_batch_offset * self.batch_size * self.num_processes
        if offset > len(indices):
            raise ValueError(f"Resume offset {offset} exceeds epoch size {len(indices)}")
        return iter(indices[offset:])

    def __len__(self) -> int:
        offset = self.resume_batch_offset * self.batch_size * self.num_processes
        return self.samples_per_epoch - offset

    def describe_sample_indices(self, indices: list[int]) -> list[dict[str, Any]]:
        descriptions = []
        for sample_index in indices:
            position = bisect.bisect_right(self._range_starts, int(sample_index)) - 1
            if position < 0:
                raise ValueError(f"Sample index is outside all episode ranges: {sample_index}")
            start, legal_stop, episode, task = self._range_records[position]
            if not start <= int(sample_index) < legal_stop:
                raise ValueError(f"Sample index is outside legal range: {sample_index}")
            descriptions.append({"task": task, "source_episode_index": episode})
        return descriptions

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler_type": "task_episode_window",
            "manifest_path": self.manifest_path,
            "selected_tasks": list(self.tasks),
            "episodes_per_task": self.episodes_per_task,
            "expected_task_count": self.expected_task_count,
            "seed": self.seed,
            "epoch": self.epoch,
            "resume_batch_offset": self.resume_batch_offset,
            "world_size": self.num_processes,
            "expected_world_size": self.expected_world_size,
            "per_rank_batch_size": self.batch_size,
            "samples_per_epoch": self.samples_per_epoch,
            "window_size": self.window_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        required = (
            "sampler_type", "manifest_path", "selected_tasks", "episodes_per_task",
            "expected_task_count", "seed", "world_size", "expected_world_size",
            "per_rank_batch_size", "samples_per_epoch", "window_size",
        )
        mismatch = {key: (state.get(key), expected[key]) for key in required if state.get(key) != expected[key]}
        if mismatch:
            raise ValueError(f"Sampler state is incompatible with current run: {mismatch}")
        self.set_epoch(int(state["epoch"]))
        self.set_resume_batch_offset(int(state["resume_batch_offset"]))

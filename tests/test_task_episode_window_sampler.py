import json

from wam_adapter.utils.samplers import TaskEpisodeWindowSampler


class _FakeRobotDataset:
    num_frames = 3

    def __len__(self):
        return 28

    def get_source_episode_sample_ranges(self):
        return {
            10: (0, 5),
            11: (5, 11),
            20: (11, 19),
            21: (19, 28),
        }


def test_task_episode_window_sampler_is_balanced_and_resumable(tmp_path):
    manifest = {
        "tasks": ["short", "long"],
        "episodes": [
            {"task_name": "short", "episode_index": 10},
            {"task_name": "short", "episode_index": 11},
            {"task_name": "long", "episode_index": 20},
            {"task_name": "long", "episode_index": 21},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sampler = TaskEpisodeWindowSampler(
        _FakeRobotDataset(),
        manifest_path=manifest_path,
        seed=42,
        batch_size=2,
        num_processes=2,
        expected_world_size=2,
        samples_per_epoch=20,
        expected_task_count=2,
        episodes_per_task=2,
    )

    indices = list(sampler)
    records = sampler.describe_sample_indices(indices)
    assert sum(record["task"] == "short" for record in records) == 10
    assert sum(record["task"] == "long" for record in records) == 10

    sampler.set_resume_batch_offset(2)
    assert list(sampler) == indices[8:]

    state = sampler.state_dict()
    restored = TaskEpisodeWindowSampler(
        _FakeRobotDataset(),
        manifest_path=manifest_path,
        seed=42,
        batch_size=2,
        num_processes=2,
        expected_world_size=2,
        samples_per_epoch=20,
        expected_task_count=2,
        episodes_per_task=2,
    )
    restored.load_state_dict(state)
    assert list(restored) == indices[8:]


def test_task_episode_window_sampler_changes_sequence_across_epochs(tmp_path):
    manifest = {
        "tasks": ["short", "long"],
        "episodes": [
            {"task_name": "short", "episode_index": 10},
            {"task_name": "short", "episode_index": 11},
            {"task_name": "long", "episode_index": 20},
            {"task_name": "long", "episode_index": 21},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sampler = TaskEpisodeWindowSampler(
        _FakeRobotDataset(),
        manifest_path=manifest_path,
        seed=42,
        batch_size=2,
        num_processes=2,
        expected_world_size=2,
        samples_per_epoch=20,
        expected_task_count=2,
        episodes_per_task=2,
    )
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero

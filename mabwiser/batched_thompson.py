from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import Arm, Num, _BaseRNG


class _BatchedThompsonSampling(BaseMAB):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[Arm],
        n_jobs: int,
        backend: Optional[str],
        batch_growth_factor: float = 1.1,
        gaussian_variance: float = 1.0,
        window_size: int = 100,
        initial_batch_size: int = 1,
        decay_factor: float = 1.0,
        binarizer: Optional[Callable] = None,
    ):
        super().__init__(rng, arms, n_jobs, backend)
        self.np_rng = np.random.default_rng(81)
        self.batch_growth_factor = batch_growth_factor
        self.gaussian_variance = gaussian_variance
        self.binarizer = binarizer
        self.window_size = window_size
        self.initial_batch_size = initial_batch_size
        self.decay_factor = decay_factor
        self.cycle_count = dict.fromkeys(self.arms, 0)
        self.cumulative_reward = dict.fromkeys(self.arms, 0.0)
        self.cycle_limit = dict.fromkeys(self.arms, initial_batch_size)
        self.batch_index = 1
        self.time_index = 1
        self.current_cycle = []
        self.arm_history = {arm: deque(maxlen=window_size) for arm in self.arms}

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        self.cycle_count = dict.fromkeys(self.arms, 0)
        self.cumulative_reward = dict.fromkeys(self.arms, 0.0)
        self.cycle_limit = dict.fromkeys(self.arms, self.initial_batch_size)
        self.batch_index = 1
        self.time_index = 1
        self.current_cycle = []
        self.arm_history = {arm: deque(maxlen=self.window_size) for arm in self.arms}

        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: np.ndarray = None) -> Arm:
        arm_samples_from_distribution = {
            arm: self.np_rng.normal(
                np.mean(self.arm_history[arm]) if self.arm_history[arm] else 0,
                np.sqrt(self.gaussian_variance / (len(self.arm_history[arm]) + 1)),
            )
            for arm in self.arms
        }
        arm_with_max_sampled_value = max(arm_samples_from_distribution, key=arm_samples_from_distribution.get)
        return arm_with_max_sampled_value

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:
        return {arm: np.mean(history) if history else 0 for arm, history in self.arm_history.items()}

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float) -> None:
        return super().warm_start(arm_to_features, distance_quantile)

    def get_current_batch_size(self) -> int:
        return max(self.cycle_limit.values(), default=0)

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[Arm, Arm]) -> None:
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.cycle_count[cold_arm] = deepcopy(self.cycle_count[warm_arm])
            self.cumulative_reward[cold_arm] = deepcopy(self.cumulative_reward[warm_arm])
            self.cycle_limit[cold_arm] = deepcopy(self.cycle_limit[warm_arm])
            self.arm_history[cold_arm] = deepcopy(self.arm_history[warm_arm])

    def _drop_existing_arm(self, arm: Arm) -> None:
        self.cycle_count.pop(arm)
        self.cumulative_reward.pop(arm)
        self.cycle_limit.pop(arm)
        self.arm_history.pop(arm)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        arm_decisions = decisions == arm
        arm_rewards = rewards[arm_decisions]

        for reward in arm_rewards:
            decayed_reward = self._apply_decay(arm, reward)
            self.arm_history[arm].append(decayed_reward)
            self.cycle_count[arm] += 1
            self.cumulative_reward[arm] += decayed_reward

        self.arm_to_expectation[arm] = np.mean(self.arm_history[arm]) if self.arm_history[arm] else 0

        self._update_cycle(arm)
        if self._is_batch_complete(arm):
            self._end_batch()

    def _apply_decay(self, arm: Arm, reward: Num) -> Num:
        decay_multiplier = self.decay_factor ** (len(self.arm_history[arm] if arm in self.arm_history else 0))
        return reward * decay_multiplier

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.cycle_count[arm] = 0
        self.cumulative_reward[arm] = 0.0
        self.cycle_limit[arm] = self.initial_batch_size
        self.arm_history[arm] = deque(maxlen=self.window_size)

    def _update_cycle(self, arm: Arm) -> None:
        if not self.current_cycle or arm != self.current_cycle[-1]:
            if len(self.current_cycle) == 2:
                self._end_cycle()
            self.current_cycle.append(arm)

    def _end_cycle(self) -> None:
        for arm in sorted(set(self.current_cycle)):
            if arm in self.cycle_count:
                self.cycle_count[arm] += 1
        self.current_cycle = []

    def _is_batch_complete(self, arm: Arm) -> bool:
        return self.cycle_count[arm] == self.cycle_limit[arm]

    def _end_batch(self) -> None:
        for arm in self.arms:
            self.cycle_limit[arm] = max(1, int(self.batch_growth_factor * self.cycle_count[arm]))
        self.batch_index += 1

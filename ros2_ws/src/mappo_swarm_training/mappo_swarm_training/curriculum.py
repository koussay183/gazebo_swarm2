"""
mappo_swarm_training.curriculum
CurriculumScheduler — track training progress and decide when to advance levels.
"""


class CurriculumScheduler:
    """
    Tracks training metrics and decides when to advance to the next curriculum level.

    Parameters
    ----------
    start_level : int
        Initial curriculum level.
    max_level : int
        Maximum curriculum level (training stops advancing beyond this).
    episodes_per_level : int
        Soft target: advance after this many episodes if criteria are met.
    success_threshold : float
        Minimum recent success rate to advance (e.g. 0.7 = 70%).
    """

    def __init__(
        self,
        start_level: int = 1,
        max_level: int = 5,
        episodes_per_level: int = 200,
        success_threshold: float = 0.7,
    ):
        self.current_level = start_level
        self.max_level = max_level
        self.episodes_per_level = episodes_per_level
        self.success_threshold = success_threshold

        self._episodes_at_level = 0
        self._total_episodes = 0

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def should_advance(
        self,
        recent_success_rate: float,
        recent_mean_reward: float = 0.0,
    ) -> bool:
        """
        Determine whether the scheduler should advance to the next level.

        Parameters
        ----------
        recent_success_rate : float
            Success rate over the last N episodes [0.0, 1.0].
        recent_mean_reward : float
            Mean reward over the last N episodes (optional, informational).

        Returns
        -------
        bool
            True if criteria are met and level < max_level.
        """
        if self.current_level >= self.max_level:
            return False
        if self._episodes_at_level < self.episodes_per_level:
            return False
        return recent_success_rate >= self.success_threshold

    def advance(self) -> int:
        """
        Advance to the next level.

        Returns
        -------
        int
            The new current level.
        """
        if self.current_level < self.max_level:
            self.current_level += 1
            self._episodes_at_level = 0
        return self.current_level

    def step(self) -> None:
        """Record that one episode has completed."""
        self._episodes_at_level += 1
        self._total_episodes += 1

    def get_current_level(self) -> int:
        """Return the current curriculum level."""
        return self.current_level

    def get_progress_string(self) -> str:
        """
        Return a human-readable progress string.

        Example: "Level 3/5 | 47/200 episodes | total 347"
        """
        return (
            f"Level {self.current_level}/{self.max_level} | "
            f"{self._episodes_at_level}/{self.episodes_per_level} episodes | "
            f"total {self._total_episodes}"
        )

    def reset_level(self) -> None:
        """Reset the episode counter for the current level."""
        self._episodes_at_level = 0

    @property
    def episodes_at_current_level(self) -> int:
        """Number of episodes completed at the current level."""
        return self._episodes_at_level

    @property
    def total_episodes(self) -> int:
        """Total number of episodes across all levels."""
        return self._total_episodes

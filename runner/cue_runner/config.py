import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RunnerConfig:
    BASE_DIR = Path(__file__).parent.parent
    RUNNERS_DIR = Path("/tmp/cue_runners")
    DEFAULT_RUNNER_ID = "default"

    @classmethod
    def get_runner_space(cls, runner_id: str) -> dict[str, Path]:
        """Get dedicated paths for a specific runner"""
        runner_dir = cls.RUNNERS_DIR / runner_id
        runner_dir.mkdir(parents=True, exist_ok=True)

        return {
            "space_dir": runner_dir,
            "control_file": runner_dir / "control.json",
            "service_file": runner_dir / "service.json",
            "pid_file": runner_dir / "runner.pid",
            "log_file": runner_dir / "runner.log",
        }

    @classmethod
    def get_all_runners(cls) -> list[str]:
        """Get list of all runner IDs, filtering empty directories"""
        if not cls.RUNNERS_DIR.exists():
            return []

        runners = []
        for runner_dir in cls.RUNNERS_DIR.iterdir():
            if runner_dir.is_dir():
                # Check if directory has any files
                if any(runner_dir.iterdir()):
                    runners.append(runner_dir.name)
                else:
                    # Clean up empty directory
                    try:
                        runner_dir.rmdir()
                        logger.info(f"Removed empty runner directory: {runner_dir}")
                    except Exception as e:
                        logger.error(f"Error removing empty directory {runner_dir}: {e}")
        return runners

    @classmethod
    def get_files(cls, runner_id: str = DEFAULT_RUNNER_ID):
        """Get all files for a specific runner"""
        space = cls.get_runner_space(runner_id)
        return [space["control_file"], space["service_file"], space["pid_file"]]

"""
Manages system self-operations like restart through control file in runner space
"""

import os
import json
import logging
from typing import Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemManagement:
    """
    Manages system self-operations like restart through control file
    """

    def __init__(self, runner_id: Optional[str] = None):
        self.runner_id = runner_id or os.environ.get("CUE_RUNNER_ID", "default")
        self.control_file = self._find_control_file()

        if self.control_file:
            logger.info(f"Using control file: {self.control_file}")

    def _find_control_file(self) -> Optional[Path]:
        """
        Find control file for this system, checking in order:
        1. Environment variable CUE_CONTROL_FILE
        2. Runner space based on CUE_RUNNER_ID
        3. Default runner space with default ID
        """
        # First check environment variable
        if env_file := os.environ.get("CUE_CONTROL_FILE"):
            control_file = Path(env_file)
            if control_file.exists():
                return control_file

        # Then check runner space
        base_dir = Path("/tmp/cue_runners")
        if self.runner_id:
            runner_dir = base_dir / self.runner_id
            control_file = runner_dir / "control.json"
            if control_file.exists():
                return control_file

        # Finally check default runner
        default_control = base_dir / "default" / "control.json"
        if default_control.exists():
            return default_control

        return None

    def _find_all_runners(self) -> dict[str, Path]:
        """Find all active runners and their control files"""
        runners = {}
        base_dir = Path("/tmp/cue_runners")

        if base_dir.exists():
            for runner_dir in base_dir.iterdir():
                if runner_dir.is_dir():
                    control_file = runner_dir / "control.json"
                    if control_file.exists():
                        try:
                            data = json.loads(control_file.read_text())
                            if data.get("status") == "running":
                                runners[runner_dir.name] = control_file
                        except (OSError, json.JSONDecodeError):
                            continue

        return runners

    async def request_restart(self, reason: str) -> bool:
        """
        Request a restart by writing to the control file.
        The CueRunner will pick up this request during its regular checks.

        Args:
            reason: Why the restart is being requested

        Returns:
            bool: True if request was written successfully
        """
        try:
            if not self.control_file or not self.control_file.exists():
                logger.error(f"Control file not found at {self.control_file}")
                return False

            # Read current control file
            data = json.loads(self.control_file.read_text())

            # Update with restart request
            data.update(
                {
                    "restart_requested": True,
                    "restart_reason": reason,
                    "restart_requested_at": datetime.now().isoformat(),
                    "status": "restart_pending",
                }
            )

            # Write back to control file
            self.control_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Restart requested for runner '{self.runner_id}': {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to request restart: {e}")
            return False

    async def get_status(self) -> Optional[dict[str, Any]]:
        """
        Get current status from control file

        Returns:
            Optional[Dict]: Current status or None if unable to read
        """
        try:
            if not self.control_file or not self.control_file.exists():
                return None

            return json.loads(self.control_file.read_text())

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return None

    async def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all active runners"""
        statuses = {}
        for runner_id, control_file in self._find_all_runners().items():
            try:
                data = json.loads(control_file.read_text())
                statuses[runner_id] = data
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read status for runner '{runner_id}': {e}")
                continue
        return statuses

    async def wait_for_restart_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for restart to complete by monitoring control file

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if restart completed, False if timed out
        """
        import asyncio

        start_time = datetime.now()
        logger.info(f"Waiting for restart completion of runner '{self.runner_id}'")

        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.get_status()
            if status and status.get("status") == "running" and not status.get("restart_requested"):
                logger.info(f"Runner '{self.runner_id}' restart completed")
                return True
            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for runner '{self.runner_id}' restart")
        return False

    async def check_health(self) -> tuple[bool, str]:
        """
        Check health of the runner

        Returns:
            Tuple[bool, str]: (is_healthy, reason)
        """
        status = await self.get_status()
        if not status:
            return False, "No status available"

        # Check if running
        if status.get("status") != "running":
            return False, f"Status is {status.get('status')}"

        # Check last check time
        try:
            last_check = datetime.fromisoformat(status.get("last_check", ""))
            if (datetime.now() - last_check).total_seconds() > 60:  # 1 minute threshold
                return False, f"Last check too old: {last_check}"
        except (ValueError, TypeError):
            return False, "Invalid last check time"

        return True, "Runner is healthy"

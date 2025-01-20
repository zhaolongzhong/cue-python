import json
import logging
from typing import Optional
from datetime import datetime, timedelta

import psutil

from .config import RunnerConfig

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(self, runner_id: str = RunnerConfig.DEFAULT_RUNNER_ID):
        self.runner_id = runner_id
        self.space = RunnerConfig.get_runner_space(runner_id)
        self.control_file = self.space["control_file"]
        self.service_file = self.space["service_file"]
        self.pid_file = self.space["pid_file"]
        self.log_file = self.space["log_file"]

    def _get_process_info(self, proc: psutil.Process, ws_info: Optional[str] = None) -> dict:
        """Get detailed process information"""
        try:
            parent = psutil.Process(proc.ppid())
            parent_cmdline = " ".join(parent.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent_cmdline = "unknown"

        cmdline = " ".join(proc.cmdline())
        runner_id = None

        # Extract runner ID from command line or environment
        try:
            if "--runner-id" in cmdline:
                parts = cmdline.split()
                idx = parts.index("--runner-id")
                if idx + 1 < len(parts):
                    runner_id = parts[idx + 1]
            else:
                runner_id = proc.environ().get("CUE_RUNNER_ID")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        proc_info = {
            "pid": proc.pid,
            "ppid": proc.ppid(),
            "parent_cmdline": parent_cmdline,
            "cmdline": cmdline,
            "runner_id": runner_id,
            "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
            "memory_info": proc.memory_info()._asdict(),
            "cpu_percent": proc.cpu_percent(),
            "status": proc.status(),
            "websocket": ws_info,
            "connections": self._get_connections(proc) if ws_info else None,
        }
        return proc_info

    def find_all_processes(self) -> list[dict]:
        """Find all running cue-related processes with parent-child relationships"""
        processes = []
        main_runner = None

        # First find all processes
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = " ".join(proc.cmdline()).lower()

                # Check if process belongs to this runner instance
                if self.runner_id != RunnerConfig.DEFAULT_RUNNER_ID:
                    if f"--runner-id {self.runner_id}" not in cmdline:
                        continue

                if "cue -r --runner" in cmdline or "run_cue.py" in cmdline:
                    proc_info = self._get_process_info(proc)

                    # Check if this is the main process for this runner
                    if self.control_file.exists():
                        control_data = json.loads(self.control_file.read_text())
                        if proc_info["pid"] == control_data.get("pid"):
                            proc_info["is_main"] = True
                            main_runner = proc_info
                        else:
                            proc_info["is_main"] = False

                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort main runner first
        if main_runner:
            processes.remove(main_runner)
            processes.insert(0, main_runner)

        return processes

    def kill_all(self, force: bool = False) -> dict[int, str]:
        """Kill all processes for this runner instance"""
        results = {}
        killed_pids = set()

        try:
            # First get the main process PID from control file
            main_pid = None
            try:
                if self.control_file.exists():
                    data = json.loads(self.control_file.read_text())
                    main_pid = data.get("pid")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error reading control file: {e}")

            # Find all processes for this runner
            target_processes = []
            for proc in psutil.process_iter(["pid", "ppid", "cmdline", "environ"]):
                try:
                    cmdline = " ".join(proc.cmdline())

                    # Check if process belongs to this runner
                    is_target = False
                    if main_pid and proc.pid == main_pid:
                        is_target = True
                    elif "--runner-id" in cmdline:
                        parts = cmdline.split()
                        try:
                            idx = parts.index("--runner-id")
                            if idx + 1 < len(parts) and parts[idx + 1] == self.runner_id:
                                is_target = True
                        except (IndexError, ValueError):
                            pass
                    elif "cue -r --runner" in cmdline:
                        try:
                            env = proc.environ()
                            if env.get("CUE_RUNNER_ID") == self.runner_id:
                                is_target = True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    if is_target:
                        target_processes.append(proc)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

            # Kill processes
            for proc in target_processes:
                try:
                    pid = proc.pid
                    if pid not in killed_pids and pid != 0:  # Explicitly check for PID 0
                        logger.info(f"Killing process {pid} for runner '{self.runner_id}'")
                        if force:
                            proc.kill()
                        else:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                proc.kill()

                        killed_pids.add(pid)
                        results[pid] = "Success"
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    results[pid] = f"Failed: {str(e)}"
                except Exception as e:
                    results[pid] = f"Error: {str(e)}"

            # Clean up files
            self._clean_files()
            return results

        except Exception as e:
            logger.error(f"Error in kill_all for runner '{self.runner_id}': {e}")
            return {"error": str(e)}

    def _clean_files(self) -> None:
        """Clean up files with better error handling"""
        for key, file in self.space.items():
            try:
                if file.exists():
                    if file.is_file():  # Only remove if it's a file
                        file.unlink(missing_ok=True)
                        logger.info(f"Removed {key} file for runner '{self.runner_id}'")
            except PermissionError as e:
                logger.warning(f"Permission error removing {key} file: {e}")
            except Exception as e:
                logger.error(f"Error removing {key} file: {e}")

    def is_runner_active(self) -> tuple[bool, str]:
        """Check if specific runner instance is active"""
        try:
            if not self.control_file.exists():
                return False, f"Control file not found for runner '{self.runner_id}'"

            data = json.loads(self.control_file.read_text())
            if data.get("runner_id") != self.runner_id:
                return (
                    False,
                    f"Control file belongs to different runner: {data.get('runner_id')}",
                )

            pid = data.get("pid")
            if not pid:
                return False, "No PID in control file"

            # Find actual running processes
            running_processes = self.find_all_processes()
            if not running_processes:
                return (
                    False,
                    f"No running processes found for runner '{self.runner_id}'",
                )

            # Check heartbeat
            last_check = datetime.fromisoformat(data.get("last_check", ""))
            if datetime.now() - last_check > timedelta(minutes=1):
                active_pids = [p["pid"] for p in running_processes]
                if pid in active_pids:
                    return True, "Runner active but heartbeat stale"
                return False, f"Runner not responding (last check: {last_check})"

            return True, "Runner is active"

        except Exception as e:
            return False, f"Error checking runner: {e}"

    def _get_connections(self, proc: psutil.Process) -> Optional[list[dict]]:
        """Get process network connections"""
        try:
            connections = proc.connections(kind="inet")
            return [
                {
                    "local_addr": f"{c.laddr.ip}:{c.laddr.port}",
                    "remote_addr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None,
                    "status": c.status,
                }
                for c in connections
            ]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def _handle_process_termination(self, proc: psutil.Process, force: bool) -> None:
        """Handle process termination with timeout"""
        if force:
            proc.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()

    def kill_one(self, pid: int, force: bool = False) -> bool:
        try:
            proc = psutil.Process(pid)
            self._handle_process_termination(proc, force)
            return True
        except psutil.NoSuchProcess:
            return False

    def get_runner_info(self) -> dict:
        """Get runner information with better empty state handling"""
        try:
            status = {
                "time_checked": datetime.now().isoformat(),
                "runner_id": self.runner_id,
                "space_dir": str(self.space["space_dir"]),
                "has_files": False,
                "is_active": False,
            }

            # Check if directory exists and has files
            if self.space["space_dir"].exists():
                status["has_files"] = any(self.space["space_dir"].iterdir())

            if status["has_files"]:
                if self.control_file.exists():
                    control_data = json.loads(self.control_file.read_text())
                    status["control_data"] = control_data
                    status["is_active"] = True

                    if pid := control_data.get("pid"):
                        try:
                            status["process_info"] = self._get_process_info(psutil.Process(pid))
                        except psutil.NoSuchProcess:
                            status["process_info"] = "Process not found"
                            status["is_active"] = False

                if self.service_file.exists():
                    status["service_data"] = json.loads(self.service_file.read_text())
            else:
                # If directory is empty, mark for cleanup
                self.clean_stale_files()

            return status
        except Exception as e:
            logger.error(f"Error getting runner info: {e}")
            return {
                "error": str(e),
                "runner_id": self.runner_id,
                "space_dir": str(self.space["space_dir"]),
            }

    def clean_stale_files(self) -> dict[str, bool]:
        """Clean up stale files and empty directories"""
        cleaned = {
            "control_file": False,
            "service_file": False,
            "pid_file": False,
            "log_file": False,
            "directory": False,
        }

        # Clean files first
        for key, file in self.space.items():
            if key == "space_dir":  # Skip directory for now
                continue
            try:
                if file.exists():
                    if file.is_file():
                        if key == "control_file":
                            try:
                                data = json.loads(file.read_text())
                                pid = data.get("pid")
                                if not pid or not psutil.pid_exists(pid) or pid == 0:
                                    file.unlink(missing_ok=True)
                                    cleaned[key] = True
                            except (json.JSONDecodeError, ValueError):
                                file.unlink(missing_ok=True)
                                cleaned[key] = True
                        else:
                            file.unlink(missing_ok=True)
                            cleaned[key] = True
                        logger.info(f"Cleaned {key} file for runner '{self.runner_id}'")
            except Exception as e:
                logger.error(f"Error cleaning {key} file: {e}")

        # Try to remove directory if empty
        try:
            space_dir = self.space["space_dir"]
            if space_dir.exists() and not any(space_dir.iterdir()):
                space_dir.rmdir()
                cleaned["directory"] = True
                logger.info(f"Removed empty directory for runner '{self.runner_id}'")
        except Exception as e:
            logger.error(f"Error removing directory: {e}")

        return cleaned

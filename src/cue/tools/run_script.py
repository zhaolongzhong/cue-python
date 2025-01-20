import os
import ast
import sys
import platform
import resource
import tempfile
import subprocess
from typing import Any, Union, Literal, ClassVar, Optional
from pathlib import Path
from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class ScriptResult:
    success: bool
    stdout: str
    stderr: str
    exception: Optional[str] = None
    exit_code: Optional[int] = None


class PythonRunner(BaseTool):
    """A secure Python script runner with sandboxing capabilities."""

    name: ClassVar[Literal["run_script"]] = "run_script"
    DEFAULT_TIMEOUT = 30
    MEMORY_LIMIT = 256 * 1024 * 1024
    MAX_FILE_SIZE = 1 * 1024 * 1024

    def __init__(
        self,
        allowed_modules: Optional[set[str]] = None,
        allowed_paths: Optional[set[str]] = None,
        timeout: Optional[int] = None,
    ):
        self.allowed_modules = allowed_modules or {
            "math",
            "random",
            "datetime",
            "json",
            "collections",
            "itertools",
            "functools",
            "time",
        }
        self.allowed_paths = allowed_paths or set()
        self.timeout = timeout or self.DEFAULT_TIMEOUT

    async def __call__(self, script: Union[str, Path], is_file: bool = False, **kwargs) -> ToolResult:
        """Execute a Python script from content or file."""
        res = await self.run_script(script, is_file)
        return self.convert_script_to_tool_result(res)

    def _validate_script_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Validate a script file before execution."""
        path = Path(file_path)

        if not path.exists():
            raise ValueError(f"Script file not found: {file_path}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"Script file exceeds maximum size of {self.MAX_FILE_SIZE} bytes")

        if self.allowed_paths and not any(str(path).startswith(str(allowed)) for allowed in self.allowed_paths):
            raise ValueError(f"Script file location not in allowed paths: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
            if len(content.encode("utf-8")) > self.MAX_FILE_SIZE:
                raise ValueError(f"Script content exceeds maximum size of {self.MAX_FILE_SIZE} bytes")
            return content
        except Exception as e:
            raise ValueError(f"Failed to read script file: {e}")

    def _validate_script_content(self, content: str) -> None:
        """Validate script content for security issues."""
        if len(content.encode("utf-8")) > self.MAX_FILE_SIZE:
            raise ValueError(f"Script content exceeds maximum size of {self.MAX_FILE_SIZE} bytes")

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        for node in ast.walk(tree):
            # Fix: Use separate isinstance checks to avoid the UP038 lint error
            if isinstance(node, ast.Import):
                module_name = node.names[0].name.split(".")[0]
                if module_name not in self.allowed_modules:
                    raise ValueError(f"Import of module '{module_name}' is not allowed")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name not in self.allowed_modules:
                        raise ValueError(f"Import of module '{module_name}' is not allowed")

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"open", "exec", "eval"}:
                        raise ValueError(f"Use of '{node.func.id}' is not allowed")

            if isinstance(node, ast.Attribute):
                if node.attr in {"system", "popen", "spawn", "fork", "exec"}:
                    raise ValueError(f"Use of '{node.attr}' is not allowed")

    def _create_sandbox_env(self) -> dict[str, Any]:
        """Create a restricted environment for script execution."""
        sandbox_env = os.environ.copy()
        sandbox_env["PYTHONPATH"] = ""
        sandbox_env.pop("PYTHONHOME", None)
        sandbox_env.pop("PYTHONSTARTUP", None)
        return sandbox_env

    def _set_resource_limits(self):
        """Set resource limits for the script execution."""
        try:
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (self.MEMORY_LIMIT, self.MEMORY_LIMIT))
            # Set maximum file size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.MAX_FILE_SIZE, self.MAX_FILE_SIZE))
        except Exception as e:
            print(f"Warning: Failed to set resource limits: {e}", file=sys.stderr)

    async def run_script(self, script: Union[str, Path], is_file: bool = False) -> ScriptResult:
        """Execute Python code in a secure sandbox environment."""
        script_path = None
        try:
            # Get and validate script content
            if is_file:
                script_content = self._validate_script_file(script)
            else:
                script_content = script

            self._validate_script_content(script_content)

            # Create temporary file for the script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                temp_file.write(script_content)
                script_path = temp_file.name

            # Create sandbox environment
            env = self._create_sandbox_env()

            # Handle resource limits based on platform
            if platform.system() != "Darwin":
                # For Linux, create a wrapper script that sets resource limits
                wrapper_script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, ({self.timeout}, {self.timeout}))
resource.setrlimit(resource.RLIMIT_AS, ({self.MEMORY_LIMIT}, {self.MEMORY_LIMIT}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({self.MAX_FILE_SIZE}, {self.MAX_FILE_SIZE}))

with open("{script_path}") as f:
    exec(f.read())
"""
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as wrapper_file:
                    wrapper_file.write(wrapper_script)
                    wrapper_path = wrapper_file.name
            else:
                wrapper_path = script_path

            try:
                # Run the script
                process = subprocess.run(
                    [sys.executable, wrapper_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=self.timeout,
                )

                if len(process.stdout.encode("utf-8")) > self.MAX_FILE_SIZE:
                    return ScriptResult(
                        success=False,
                        stdout="",
                        stderr=f"Script output exceeds maximum size of {self.MAX_FILE_SIZE} bytes",
                        exit_code=1,
                    )

                return ScriptResult(
                    success=process.returncode == 0,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    exit_code=process.returncode,
                )

            except subprocess.TimeoutExpired:
                return ScriptResult(success=False, stdout="", stderr="Script execution timed out", exit_code=124)
            finally:
                if platform.system() != "Darwin" and os.path.exists(wrapper_path):
                    try:
                        os.unlink(wrapper_path)
                    except OSError:
                        pass

        except ValueError as e:
            return ScriptResult(success=False, stdout="", stderr=str(e), exception=str(e), exit_code=1)
        except Exception as e:
            return ScriptResult(success=False, stdout="", stderr=str(e), exception=str(e), exit_code=1)
        finally:
            if script_path and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

    def convert_script_to_tool_result(self, script_result: ScriptResult) -> ToolResult:
        """
        Convert a ScriptResult instance to a ToolResult instance.

        The conversion logic:
        - If success is True, stdout goes to output
        - If success is False, stderr goes to error
        - If exception exists, it's added to error (with stderr if present)
        - System field gets exit_code if present
        - base64_image is always None as ScriptResult doesn't have equivalent
        """
        error = None
        if not script_result.success:
            error_parts = []
            if script_result.stderr:
                error_parts.append(script_result.stderr)
            if script_result.exception:
                error_parts.append(f"Exception: {script_result.exception}")
            error = "\n".join(error_parts) if error_parts else None

        system = str(script_result.exit_code) if script_result.exit_code is not None else None

        return ToolResult(
            output=script_result.stdout if script_result.success else None,
            error=error,
            system=system,
            base64_image=None,
        )

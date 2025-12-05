# import subprocess
# import os
# from dataclasses import dataclass
# from typing import Optional

# @dataclass
# class ExecResult:
#     returncode: int
#     stdout: str
#     stderr: str
#     timed_out: bool

# class Executor:
#     def __init__(self, cwd: str, timeout: int = 10, verbose: bool=False):
#         self.cwd = os.path.realpath(cwd)
#         self.timeout = timeout
#         self.verbose = verbose

#     def run_command(self, cmd: str) -> ExecResult:
#         """
#         Runs the provided shell command in the project cwd.
#         WARNING: This executes arbitrary commands. Use in trusted environments.
#         """
#         try:
#             proc = subprocess.run(cmd, shell=True, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout)
#             return ExecResult(proc.returncode, proc.stdout, proc.stderr, timed_out=False)
#         except subprocess.TimeoutExpired as e:
#             return ExecResult(-1, getattr(e, "output", "") or "", getattr(e, "stderr", "") or "Timed out", timed_out=True)
#         except Exception as e:
#             return ExecResult(-1, "", str(e), timed_out=False)

import shlex
from typing import List, Union

class Executor:
    ALLOWED_COMMANDS = {'pytest', 'python', 'pip', 'npm', 'node', 'cargo'}
    
    def __init__(self, cwd: str, timeout: int = 60, verbose: bool = False):
        self.cwd = os.path.realpath(cwd)
        self.timeout = timeout
        self.verbose = verbose
        self._validate_cwd()
    
    def _validate_cwd(self):
        if not os.path.isdir(self.cwd):
            raise ValueError(f"Invalid working directory: {self.cwd}")
    
    def _sanitize_command(self, cmd: str) -> List[str]:
        """Convert command string to safe argument list."""
        try:
            args = shlex.split(cmd)
            if not args:
                raise ValueError("Empty command")
            
            # Optional: Whitelist check
            base_cmd = args[0].split('/')[-1]
            if base_cmd not in self.ALLOWED_COMMANDS:
                raise ValueError(f"Command not allowed: {base_cmd}")
            
            return args
        except Exception as e:
            raise ValueError(f"Invalid command: {e}")
    
    def run_command(self, cmd: str, allow_shell: bool = False) -> ExecResult:
        """
        Runs command safely. Use allow_shell=True only for trusted commands.
        """
        if self.verbose:
            print(f"[Executor] Running: {cmd}")
        
        try:
            if allow_shell:
                # Only for trusted contexts (with warning)
                proc = subprocess.run(
                    cmd, 
                    shell=True, 
                    cwd=self.cwd, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.timeout
                )
            else:
                # Safer: Parse into arguments
                args = self._sanitize_command(cmd)
                proc = subprocess.run(
                    args,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            
            return ExecResult(proc.returncode, proc.stdout, proc.stderr, False)
        
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                -1, 
                e.output.decode() if e.output else "", 
                e.stderr.decode() if e.stderr else "Timed out", 
                True
            )
        except Exception as e:
            return ExecResult(-1, "", f"Execution error: {e}", False)
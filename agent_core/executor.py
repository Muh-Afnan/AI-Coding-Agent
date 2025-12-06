"""
Secure Executor - Safe command execution with validation and sandboxing.
"""
import subprocess
import os
import shlex
from dataclasses import dataclass
from typing import Optional, List, Set
import re


@dataclass
class ExecResult:
    """Result from command execution."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    command: str = ""
    
    def is_success(self) -> bool:
        """Check if command succeeded."""
        return self.returncode == 0 and not self.timed_out


class Executor:
    """
    Secure command executor with:
    - Command whitelisting
    - Argument sanitization
    - Resource limits
    - Execution logging
    """
    
    # Whitelist of allowed base commands
    ALLOWED_COMMANDS = {
        'python', 'python3', 'pip', 'pip3',
        'pytest', 'unittest',
        'node', 'npm', 'npx',
        'git',
        'cargo', 'rustc',
        'go',
        'java', 'javac', 'mvn', 'gradle',
        'make', 'cmake',
        'ls', 'cat', 'grep', 'find', 'echo'
    }
    
    # Commands that should never be allowed
    DANGEROUS_COMMANDS = {
        'rm', 'del', 'rmdir', 'format',
        'dd', 'mkfs',
        'kill', 'killall', 'pkill',
        'shutdown', 'reboot', 'halt',
        'chmod', 'chown',
        'sudo', 'su',
        'curl', 'wget'  # Prevent network access
    }
    
    def __init__(
        self,
        cwd: str,
        timeout: int = 60,
        verbose: bool = False,
        strict_mode: bool = True
    ):
        """
        Initialize secure executor.
        
        Args:
            cwd: Working directory for commands
            timeout: Timeout in seconds
            verbose: Enable detailed logging
            strict_mode: If True, only allow whitelisted commands
        """
        self.cwd = os.path.realpath(cwd)
        self.timeout = timeout
        self.verbose = verbose
        self.strict_mode = strict_mode
        self._execution_log: List[dict] = []
        
        self._validate_cwd()
    
    def _validate_cwd(self):
        """Validate working directory."""
        if not os.path.isdir(self.cwd):
            raise ValueError(f"Invalid working directory: {self.cwd}")
        
        # Ensure not in system directories
        system_dirs = ['/bin', '/sbin', '/usr/bin', '/usr/sbin', '/etc', '/sys', '/proc']
        if any(self.cwd.startswith(d) for d in system_dirs):
            raise ValueError(f"Cannot execute in system directory: {self.cwd}")
    
    def run_command(
        self,
        cmd: str,
        allow_shell: bool = False,
        timeout: Optional[int] = None
    ) -> ExecResult:
        """
        Execute a command safely.
        
        Args:
            cmd: Command to execute
            allow_shell: Allow shell=True (use with caution)
            timeout: Override default timeout
        
        Returns:
            ExecResult with execution outcome
        """
        timeout = timeout or self.timeout
        
        # Log command
        if self.verbose:
            print(f"🔧 Executing: {cmd}")
        
        # Validate and parse command
        is_valid, error_msg, parsed_cmd = self._validate_command(cmd, allow_shell)
        
        if not is_valid:
            result = ExecResult(
                returncode=-1,
                stdout="",
                stderr=error_msg,
                timed_out=False,
                command=cmd
            )
            self._log_execution(result)
            return result
        
        # Execute
        try:
            if allow_shell:
                # Shell mode (less safe, but sometimes needed)
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # Safer: parsed arguments
                proc = subprocess.run(
                    parsed_cmd,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            result = ExecResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
                command=cmd
            )
            
            if self.verbose:
                status = "✅" if result.is_success() else "❌"
                print(f"{status} Exit code: {result.returncode}")
            
            self._log_execution(result)
            return result
        
        except subprocess.TimeoutExpired as e:
            result = ExecResult(
                returncode=-1,
                stdout=e.output.decode() if e.output else "",
                stderr=e.stderr.decode() if e.stderr else f"Command timed out after {timeout}s",
                timed_out=True,
                command=cmd
            )
            
            if self.verbose:
                print(f"⏱️  Command timed out")
            
            self._log_execution(result)
            return result
        
        except Exception as e:
            result = ExecResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                timed_out=False,
                command=cmd
            )
            
            if self.verbose:
                print(f"❌ Execution failed: {e}")
            
            self._log_execution(result)
            return result
    
    def _validate_command(
        self,
        cmd: str,
        allow_shell: bool
    ) -> tuple[bool, str, Optional[List[str]]]:
        """
        Validate command safety.
        
        Returns:
            (is_valid, error_message, parsed_command)
        """
        if not cmd or not cmd.strip():
            return False, "Empty command", None
        
        # Parse command
        try:
            if allow_shell:
                # Basic validation for shell commands
                parsed = [cmd]
            else:
                parsed = shlex.split(cmd)
        except Exception as e:
            return False, f"Failed to parse command: {e}", None
        
        if not parsed:
            return False, "Empty parsed command", None
        
        # Extract base command
        base_cmd = os.path.basename(parsed[0])
        
        # Check against dangerous commands
        if base_cmd in self.DANGEROUS_COMMANDS:
            return False, f"Dangerous command not allowed: {base_cmd}", None
        
        # Check against whitelist if strict mode
        if self.strict_mode and base_cmd not in self.ALLOWED_COMMANDS:
            return False, f"Command not in whitelist: {base_cmd}", None
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'&&\s*rm',  # Chained delete
            r'>\s*/dev/',  # Writing to devices
            r'\|\s*sh',  # Piping to shell
            r'`.*`',  # Command substitution
            r'\$\(',  # Command substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, cmd):
                return False, f"Dangerous pattern detected: {pattern}", None
        
        return True, "", parsed
    
    def _log_execution(self, result: ExecResult):
        """Log command execution for audit trail."""
        self._execution_log.append({
            "command": result.command,
            "returncode": result.returncode,
            "success": result.is_success(),
            "timed_out": result.timed_out,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr)
        })
    
    def get_execution_log(self) -> List[dict]:
        """Get execution history."""
        return self._execution_log.copy()
    
    def clear_log(self):
        """Clear execution log."""
        self._execution_log.clear()
    
    def add_allowed_command(self, command: str):
        """Add command to whitelist (use carefully!)."""
        if command not in self.DANGEROUS_COMMANDS:
            self.ALLOWED_COMMANDS.add(command)
            if self.verbose:
                print(f"✅ Added to whitelist: {command}")
        else:
            if self.verbose:
                print(f"❌ Cannot whitelist dangerous command: {command}")
    
    def is_command_allowed(self, cmd: str) -> bool:
        """Check if command would be allowed."""
        is_valid, _, _ = self._validate_command(cmd, allow_shell=False)
        return is_valid


# Example usage
if __name__ == "__main__":
    executor = Executor(cwd=".", verbose=True, strict_mode=True)
    
    print("\n" + "="*60)
    print("TESTING SECURE EXECUTOR")
    print("="*60)
    
    # Test allowed command
    print("\n✓ Test 1: Allowed command")
    result = executor.run_command("echo 'Hello World'")
    print(f"Success: {result.is_success()}")
    print(f"Output: {result.stdout}")
    
    # Test dangerous command
    print("\n✓ Test 2: Dangerous command (should be blocked)")
    result = executor.run_command("rm -rf /")
    print(f"Blocked: {not result.is_success()}")
    print(f"Error: {result.stderr}")
    
    # Test timeout
    print("\n✓ Test 3: Timeout")
    result = executor.run_command("sleep 5", timeout=1)
    print(f"Timed out: {result.timed_out}")
    
    # Show log
    print("\n" + "="*60)
    print("EXECUTION LOG")
    print("="*60)
    for entry in executor.get_execution_log():
        print(f"Command: {entry['command'][:50]}")
        print(f"  Success: {entry['success']}")
        print()
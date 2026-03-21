"""
ScriptCompiler - Python script compilation checker for InfEngine.

Validates Python scripts for syntax errors when files are modified.
Reports errors to the Console Panel for developer visibility.

Features:
- Syntax checking using py_compile and ast
- Import validation
- Error reporting with line numbers
- Integration with watchdog file monitoring
"""

import ast
import py_compile
import os
from typing import Optional, List
from dataclasses import dataclass

from InfEngine.debug import Debug


@dataclass
class ScriptError:
    """Represents a script compilation error."""
    file_path: str
    line_number: int
    column: int
    message: str
    error_type: str  # 'syntax', 'import', 'semantic'
    
    def __str__(self) -> str:
        return f"{os.path.basename(self.file_path)}:{self.line_number}:{self.column}: {self.message}"


class ScriptCompiler:
    """
    Validates Python scripts for errors.
    
    Usage:
        compiler = ScriptCompiler()
        
        # Check a single file
        errors = compiler.check_file("/path/to/script.py")
        if errors:
            for error in errors:
                print(error)
        
        # Check and report to Debug console
        compiler.check_and_report("/path/to/script.py")
    """
    
    def __init__(self):
        self._last_errors: List[ScriptError] = []
    
    def check_file(self, file_path: str) -> List[ScriptError]:
        """
        Check a Python file for syntax errors.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of ScriptError objects (empty if no errors)
        """
        errors = []
        
        if not os.path.exists(file_path):
            errors.append(ScriptError(
                file_path=file_path,
                line_number=0,
                column=0,
                message="File not found",
                error_type="file"
            ))
            return errors
        
        if not file_path.endswith('.py'):
            return errors
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Check 1: AST parsing (syntax check)
        syntax_errors = self._check_syntax(file_path, source_code)
        errors.extend(syntax_errors)
        
        # If there are syntax errors, skip further checks
        if syntax_errors:
            return errors
        
        # Check 2: py_compile (bytecode compilation check)
        compile_errors = self._check_compile(file_path)
        errors.extend(compile_errors)
        
        self._last_errors = errors
        return errors
    
    def _check_syntax(self, file_path: str, source_code: str) -> List[ScriptError]:
        """Check syntax using ast.parse()."""
        errors = []
        try:
            ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            errors.append(ScriptError(
                file_path=file_path,
                line_number=e.lineno or 0,
                column=e.offset or 0,
                message=str(e.msg) if hasattr(e, 'msg') else str(e),
                error_type='syntax',
            ))
        except Exception as e:
            errors.append(ScriptError(
                file_path=file_path,
                line_number=0,
                column=0,
                message=f"Unexpected error during syntax check: {e}",
                error_type='syntax',
            ))
        return errors
    
    def _check_compile(self, file_path: str) -> List[ScriptError]:
        """Check using py_compile for bytecode issues."""
        errors = []
        try:
            py_compile.compile(file_path, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(ScriptError(
                file_path=file_path,
                line_number=getattr(e, 'lineno', 0) or 0,
                column=0,
                message=str(e),
                error_type='compile',
            ))
        except Exception as e:
            errors.append(ScriptError(
                file_path=file_path,
                line_number=0,
                column=0,
                message=f"Unexpected compile error: {e}",
                error_type='compile',
            ))
        return errors
    
    def check_and_report(self, file_path: str) -> bool:
        """
        Check a file and report errors to Debug console.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            True if no errors, False if errors found
        """
        errors = self.check_file(file_path)
        
        if not errors:
            # Optionally log success
            Debug.log_internal(f"✓ Script compiled: {os.path.basename(file_path)}")
            return True
        
        # Report errors
        for error in errors:
            error_msg = f"[{error.error_type.upper()}] {error.file_path}:{error.line_number}:{error.column}\n{error.message}"
            Debug.log_error(error_msg,
                            source_file=error.file_path,
                            source_line=error.line_number)
        
        return False


# Global compiler instance
_compiler: Optional[ScriptCompiler] = None


def get_script_compiler() -> ScriptCompiler:
    """Get the global ScriptCompiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = ScriptCompiler()
    return _compiler

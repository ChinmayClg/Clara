import os
import subprocess
import difflib
from pathlib import Path
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SystemController:
    @staticmethod
    def _normalize_path_prefix(file_path: str) -> str:
        """
        Strip leading 'Desktop/' or 'Documents/' prefix and resolve to actual path.
        Prevents double-prefix paths like Desktop/Desktop/...
        """
        import re
        # Match Desktop/ or Documents/ at the start (case-insensitive)
        match = re.match(r'^(Desktop|Documents)[/\\](.+)$', file_path, re.IGNORECASE)
        if match:
            folder_name = match.group(1)
            rest = match.group(2)
            return str(Path.home() / folder_name / rest)
        return file_path

    def _resolve_path(self, file_path: str) -> Path:
        """
        Intelligently resolve file paths by trying multiple strategies.
        Returns resolved Path object or raises FileNotFoundError with helpful message.
        """
        # Normalize Desktop/Documents prefix first
        file_path = self._normalize_path_prefix(file_path)
        
        # Try as-is first
        path = Path(file_path)
        if path.exists():
            return path.resolve()
        
        # Try common base directories
        base_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.cwd(),
        ]
        
        attempted_paths = [str(path)]
        
        for base_dir in base_dirs:
            # Try relative to base directory
            candidate = base_dir / file_path
            attempted_paths.append(str(candidate))
            if candidate.exists():
                logger.info(f"Resolved '{file_path}' to '{candidate}'")
                return candidate.resolve()
            
            # Try without leading slash/backslash
            cleaned = file_path.lstrip('/\\').lstrip()
            candidate = base_dir / cleaned
            if candidate.exists() and str(candidate) not in attempted_paths:
                attempted_paths.append(str(candidate))
                logger.info(f"Resolved '{file_path}' to '{candidate}'")
                return candidate.resolve()
            
            # Special case: if path starts with "Desktop/", strip it and try from Desktop
            if file_path.startswith("Desktop/") or file_path.startswith("Desktop\\"):
                stripped = file_path.split("/", 1)[1] if "/" in file_path else file_path.split("\\", 1)[1]
                candidate = Path.home() / "Desktop" / stripped
                if candidate.exists() and str(candidate) not in attempted_paths:
                    attempted_paths.append(str(candidate))
                    logger.info(f"Resolved '{file_path}' to '{candidate}'")
                    return candidate.resolve()
        
        # Path not found - provide helpful error
        error_msg = f"File not found: '{file_path}'\nTried locations:\n"
        error_msg += "\n".join(f"  - {p}" for p in attempted_paths[:5])
        logger.warning(error_msg)
        raise FileNotFoundError(error_msg)

    def open_app(self, app_name: str):
        """
        Opens a desktop application on Windows.
        
        Args:
            app_name: The name of the application to open (e.g., 'calculator', 'notepad', 'browser', 'settings', 'camera').
        """
        # Clean the input slightly
        app_name = app_name.lower().replace("the ", "").replace("a ", "").strip()
        
        apps = {
            "calculator": "calc",
            "notepad": "notepad",
            "cmd": "cmd",
            "command prompt": "cmd",
            "terminal": "cmd",
            "browser": "start msedge",
            "chrome": "start chrome",
            "edge": "start msedge",
            "camera": "start microsoft.windows.camera:",
            "settings": "start ms-settings:",
            "explorer": "explorer",
            "files": "explorer",
            "file manager": "explorer",
            "vscode": "code",
            "code": "code",
            "spotify": "start spotify:",
            "whatsapp": "start whatsapp:"
        }
        
        cmd = apps.get(app_name)
        
        # If no exact match, try fuzzy match
        if not cmd:
            matches = difflib.get_close_matches(app_name, apps.keys(), n=1, cutoff=0.6)
            if matches:
                logger.info(f"Fuzzy matching '{app_name}' to '{matches[0]}'")
                cmd = apps[matches[0]]
        
        try:
            if cmd:
                subprocess.Popen(cmd, shell=True)
                logger.info(f"Opened {app_name} (mapped to {cmd})")
                return f"Opening {app_name} (mapped to {cmd})..."
            else:
                # Fallback: Dynamic Start Menu Search via PowerShell
                # This searches for .lnk files in the Start Menu matching the app name
                logger.debug(f"Searching Start Menu for '{app_name}'...")
                ps_script = f"""
                $limit = 1
                $term = "*{app_name}*"
                $paths = @(
                    "$env:ProgramData\\Microsoft\\Windows\\Start Menu\\Programs",
                    "$env:AppData\\Microsoft\\Windows\\Start Menu\\Programs"
                )
                Get-ChildItem -Path $paths -Recurse -Include *.lnk, *.url | 
                Where-Object {{ $_.Name -like $term }} | 
                Select-Object -ExpandProperty FullName -First 1
                """
                
                # Execute PowerShell to find the shortcut
                process = subprocess.Popen(["powershell", "-Command", ps_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, _ = process.communicate()
                found_path = stdout.strip()
                
                if found_path:
                    # Execute the found shortcut
                    subprocess.Popen(f'start "" "{found_path}"', shell=True)
                    logger.info(f"Found and opening: {found_path}")
                    return f"Found and opening: {found_path}"
                else:
                    # Final fallback: Generic start command
                    subprocess.Popen(f'start "" "{app_name}"', shell=True)
                    return f"Attempting generic open for '{app_name}' (might fail if not in PATH)..."
                    
        except Exception as e:
            logger.error(f"Failed to open {app_name}: {e}")
            return f"Failed to open {app_name}: {e}"

    def open_url(self, url: str):
        """
        Opens a URL in the default browser.
        Args:
            url: The URL to open (e.g., 'https://youtube.com', 'google.com').
        """
        url = url.strip()
        if not url:
            return "Error: No URL provided."
        
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            # cmd /c start ... handles URLs nicely on Windows
            subprocess.Popen(f'start "" "{url}"', shell=True)
            logger.info(f"Opening URL: {url}")
            return f"Opening URL: {url}"
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {e}")
            return f"Failed to open URL {url}: {e}"

    def search_web(self, query: str):
        """
        Searches the web for the given query using the default browser.
        """
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        return self.open_url(url)

    def read_file(self, file_path: str):
        """
        Read and return file contents with intelligent path resolution.
        """
        try:
            # Resolve path intelligently
            resolved_path = self._resolve_path(file_path)
            
            # Safety check
            if not Config.is_safe_path(str(resolved_path)):
                logger.warning(f"Attempted to read file outside safe directories: {resolved_path}")
                return f"Error: Cannot read files outside allowed directories: {resolved_path}"
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read file: {resolved_path} ({len(content)} chars)")
            return content
        except FileNotFoundError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return f"Failed to read file {file_path}: {e}"
    
    def write_file(self, file_path: str, content: str):
        """
        Writes content to a file with intelligent path resolution. Overwrites if it exists.
        """
        try:
            # Normalize Desktop/Documents prefix first
            file_path = self._normalize_path_prefix(file_path)
            
            # Try to resolve the path (for existing files) or use parent directory resolution
            path = Path(file_path)
            
            # If file doesn't exist, try to resolve parent directory
            if not path.exists() and not path.is_absolute():
                try:
                    # Default to Desktop (not cwd) when parent is '.' — prevents writing into the Assistant folder
                    default_base = Path.home() / "Desktop"
                    parent_dir = self._resolve_path(str(path.parent)) if str(path.parent) != '.' else default_base
                    resolved_path = parent_dir / path.name
                except FileNotFoundError:
                    # Parent doesn't exist, default to Desktop
                    resolved_path = Path.home() / "Desktop" / path
            else:
                resolved_path = path
            
            # Safety check
            if not Config.is_safe_path(str(resolved_path)):
                logger.warning(f"Attempted to write file outside safe directories: {resolved_path}")
                return f"Error: Cannot write files outside allowed directories: {resolved_path}"
            
            # Ensure directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify the write succeeded and report actual size
            actual_size = resolved_path.stat().st_size
            logger.info(f"Wrote to file: {resolved_path} ({actual_size} bytes, {len(content)} chars)")
            return f"Successfully wrote to {resolved_path} ({actual_size} bytes)"
        except PermissionError:
            logger.error(f"Permission denied writing to {file_path}")
            return f"Error: Permission denied — cannot write to {file_path}"
        except OSError as e:
            logger.error(f"OS error writing file {file_path}: {e}")
            return f"Error: Failed to write file {file_path}: {e}"
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return f"Failed to write file {file_path}: {e}"
    
    def create_folder(self, folder_path: str):
        """
        Creates a folder (and any parent folders) at the given path.
        Uses intelligent path resolution like other file operations.
        """
        try:
            # Normalize Desktop/Documents prefix first
            folder_path = self._normalize_path_prefix(folder_path)
            path = Path(folder_path)
            
            # If not absolute, try to place it relative to Desktop first
            if not path.is_absolute():
                base_dirs = [
                    Path.home() / "Desktop",
                    Path.home() / "Documents",
                    # NOTE: Path.cwd() intentionally excluded — prevents creating folders
                    # inside the Assistant project when the user means Desktop.
                ]
                # Check if the parent already exists somewhere
                for base_dir in base_dirs:
                    candidate = base_dir / folder_path
                    if candidate.parent.exists():
                        path = candidate
                        break
                else:
                    # Default to Desktop
                    path = Path.home() / "Desktop" / folder_path
            
            # Safety check
            if not Config.is_safe_path(str(path)):
                logger.warning(f"Attempted to create folder outside safe directories: {path}")
                return f"Error: Cannot create folders outside allowed directories: {path}"
            
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created folder: {path}")
            return f"Successfully created folder: {path}"
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return f"Failed to create folder {folder_path}: {e}"
    
    def list_dir(self, dir_path: str = "."):
        """
        Lists the contents of a directory.
        Returns file/folder names with indicators (📁 for folders, 📄 for files).
        """
        try:
            # Try to resolve the path
            try:
                resolved_path = self._resolve_path(dir_path)
            except FileNotFoundError:
                # Try common directories
                for base in [Path.home() / "Desktop", Path.home() / "Documents", Path.cwd()]:
                    candidate = base / dir_path
                    if candidate.exists():
                        resolved_path = candidate
                        break
                else:
                    return f"Directory not found: {dir_path}"
            
            if not resolved_path.is_dir():
                return f"Not a directory: {resolved_path}"
            
            items = []
            for item in sorted(resolved_path.iterdir()):
                # Skip hidden and ignored directories
                if item.name.startswith('.') or item.name in Config.IGNORE_DIRS:
                    continue
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    items.append(f"📄 {item.name} ({size_str})")
            
            if not items:
                return f"Directory is empty: {resolved_path}"
            
            header = f"Contents of {resolved_path}:\n"
            result = header + "\n".join(items)
            logger.info(f"Listed directory: {resolved_path} ({len(items)} items)")
            return result
        except Exception as e:
            logger.error(f"Failed to list directory {dir_path}: {e}")
            return f"Failed to list directory {dir_path}: {e}"
    
    def run_command(self, command: str):
        """
        Runs a shell command and returns the output.
        Restricted to safe commands (no delete/format/shutdown).
        Supports 'cd <path> && <command>' syntax for CWD control.
        """
        try:
            # Safety check — block dangerous commands
            cmd_lower = command.lower().strip()
            for keyword in Config.DANGEROUS_KEYWORDS:
                if keyword in cmd_lower:
                    logger.warning(f"Blocked dangerous command: {command}")
                    return f"Error: Command blocked for safety — contains '{keyword}'. Dangerous commands are not allowed."
            
            # Additional blocks for especially dangerous patterns
            dangerous_patterns = ['rm -rf', 'rmdir /s', 'del /f', 'format ', 'shutdown', 'taskkill']
            for pattern in dangerous_patterns:
                if pattern in cmd_lower:
                    logger.warning(f"Blocked dangerous command pattern: {command}")
                    return f"Error: Command blocked — '{pattern}' is not allowed."
            
            # --- Parse 'cd <path> && <actual command>' ---
            import re as _re
            cwd = str(Path.home() / "Desktop")  # default
            actual_command = command
            
            cd_match = _re.match(r'^cd\s+([^&]+?)\s*&&\s*(.+)$', command.strip(), _re.IGNORECASE)
            if cd_match:
                raw_cd_path = cd_match.group(1).strip().strip('"').strip("'")
                actual_command = cd_match.group(2).strip()
                
                # Resolve cd path — support relative paths like 'Desktop\spotify-clone'
                cd_path = Path(raw_cd_path)
                if not cd_path.is_absolute():
                    for base in [Path.home() / "Desktop", Path.home() / "Documents", Path.home()]:
                        candidate = base / raw_cd_path
                        if candidate.exists() and candidate.is_dir():
                            cd_path = candidate
                            break
                    else:
                        cd_path = Path.home() / "Desktop" / raw_cd_path
                
                if cd_path.exists() and cd_path.is_dir():
                    cwd = str(cd_path)
                    logger.info(f"[run_command] cwd set to: {cwd}")
                else:
                    logger.warning(f"[run_command] cd target not found: {raw_cd_path}, using Desktop")
            
            logger.info(f"Running command: {actual_command} (cwd={cwd})")
            
            process = subprocess.Popen(
                actual_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
            
            try:
                # npx create-react-app can take a long time, so use 300 seconds (5 mins)
                stdout, stderr = process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                return "Error: Command timed out after 300 seconds."
            
            output = ""
            if stdout.strip():
                output += stdout.strip()
            if stderr.strip():
                output += f"\nSTDERR: {stderr.strip()}"
            if process.returncode != 0:
                output += f"\n(Exit code: {process.returncode})"
            
            if not output.strip():
                output = f"Command completed successfully (exit code {process.returncode})."
            
            # Smart truncation: keep first 500 + last 1500 chars
            # so errors at the end of output aren't lost
            if len(output) > 3000:
                head = output[:500]
                tail = output[-1500:]
                output = head + "\n\n...[OUTPUT TRUNCATED — showing first 500 + last 1500 chars]...\n\n" + tail
            
            logger.info(f"Command output ({len(output)} chars): {output[:100]}...")
            return output
        except Exception as e:
            logger.error(f"Failed to run command '{command}': {e}")
            return f"Failed to run command: {e}"

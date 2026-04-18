import os
import re
from pathlib import Path
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CodeContext:
    """Smart code context gatherer — prioritizes high-signal files, extracts API endpoints,
    and generates project structure trees for better LLM understanding."""
    
    # File priority tiers (higher priority = read first, never skipped)
    PRIORITY_PATTERNS = {
        1: ['route', 'router', 'controller', 'api', 'endpoint', 'view', 'handler', 'urls'],
        2: ['model', 'schema', 'entity', 'type', 'interface', 'dto'],
        3: ['service', 'middleware', 'auth', 'guard', 'interceptor', 'pipe'],
        4: ['util', 'helper', 'config', 'constant', 'env'],
    }
    
    # Regex patterns to extract API endpoints from various frameworks
    ENDPOINT_PATTERNS = [
        # Express.js: app.get("/path", ...) or router.post("/path", ...)
        re.compile(r'(?:app|router)\s*\.\s*(get|post|put|patch|delete)\s*\(\s*[\'"]([^\'"]+)[\'"]', re.IGNORECASE),
        # Flask / FastAPI decorators: @app.route("/path") or @router.get("/path")
        re.compile(r'@\s*(?:app|router|bp|blueprint)\s*\.\s*(?:route\s*\(\s*[\'"]([^\'"]+)[\'"](?:.*?methods\s*=\s*\[([^\]]+)\])?|'
                   r'(get|post|put|patch|delete)\s*\(\s*[\'"]([^\'"]+)[\'"])', re.IGNORECASE),
        # Django urls: path("url/", view)
        re.compile(r'path\s*\(\s*[\'"]([^\'"]+)[\'"]', re.IGNORECASE),
        # Generic REST patterns in comments or strings
        re.compile(r'(?:GET|POST|PUT|PATCH|DELETE)\s+(/\S+)', re.IGNORECASE),
    ]
    
    # Patterns to extract model/schema names
    MODEL_PATTERNS = [
        re.compile(r'class\s+(\w+)\s*\(.*?(?:Model|Schema|Base|Entity|Document)', re.IGNORECASE),
        re.compile(r'(?:const|let|var)\s+(\w+)Schema\s*=', re.IGNORECASE),
        re.compile(r'mongoose\.(?:model|Schema)\s*\(\s*[\'"](\w+)[\'"]', re.IGNORECASE),
        re.compile(r'(?:interface|type)\s+(\w+)\s*(?:extends|{)', re.IGNORECASE),
    ]
    
    def _get_file_priority(self, filename: str) -> int:
        """Assign priority to a file based on its name. Lower = higher priority."""
        name_lower = filename.lower()
        for priority, patterns in self.PRIORITY_PATTERNS.items():
            if any(p in name_lower for p in patterns):
                return priority
        return 5  # Default/lowest priority
    
    def _build_project_tree(self, root_dir: str, max_depth: int = 4) -> str:
        """Generate a clean directory tree showing project structure."""
        tree_lines = []
        root_path = Path(root_dir)
        tree_lines.append(f"📁 {root_path.name}/")
        
        ignore_dirs = Config.IGNORE_DIRS
        
        def _walk(current_dir: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                entries = sorted(current_dir.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return
            
            # Filter entries
            filtered = []
            for entry in entries:
                if entry.name.startswith('.'):
                    continue
                if entry.is_dir() and entry.name in ignore_dirs:
                    continue
                filtered.append(entry)
            
            for i, entry in enumerate(filtered):
                is_last = (i == len(filtered) - 1)
                connector = "└── " if is_last else "├── "
                
                if entry.is_dir():
                    tree_lines.append(f"{prefix}{connector}📁 {entry.name}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _walk(entry, new_prefix, depth + 1)
                else:
                    ext = entry.suffix.lower()
                    if ext not in Config.IGNORE_EXTS:
                        size = entry.stat().st_size
                        size_str = f"{size}B" if size < 1024 else f"{size // 1024}KB"
                        tree_lines.append(f"{prefix}{connector}📄 {entry.name} ({size_str})")
        
        _walk(root_path, "", 0)
        return "\n".join(tree_lines)
    
    def _extract_api_endpoints(self, file_path: str, content: str) -> list:
        """Extract API endpoint definitions from a file."""
        endpoints = []
        rel_path = os.path.basename(file_path)
        
        for line_num, line in enumerate(content.split('\n'), 1):
            # Express/Koa style: app.get("/path", ...)
            match = re.search(r'(?:app|router)\s*\.\s*(get|post|put|patch|delete)\s*\(\s*[\'"]([^\'"]+)[\'"]', line, re.IGNORECASE)
            if match:
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append(f"  - {method} {path} ({rel_path}:{line_num})")
                continue
            
            # Flask/FastAPI decorator style
            match = re.search(r'@\s*\w+\.\s*(?:route\s*\(\s*[\'"]([^\'"]+)[\'"]|'
                              r'(get|post|put|patch|delete)\s*\(\s*[\'"]([^\'"]+)[\'"])', line, re.IGNORECASE)
            if match:
                if match.group(1):  # @app.route("/path")
                    path = match.group(1)
                    endpoints.append(f"  - ROUTE {path} ({rel_path}:{line_num})")
                elif match.group(2) and match.group(3):  # @app.get("/path")
                    method = match.group(2).upper()
                    path = match.group(3)
                    endpoints.append(f"  - {method} {path} ({rel_path}:{line_num})")
                continue
            
            # Django path()
            match = re.search(r'path\s*\(\s*[\'"]([^\'"]+)[\'"]', line, re.IGNORECASE)
            if match:
                path = match.group(1)
                endpoints.append(f"  - PATH /{path} ({rel_path}:{line_num})")
        
        return endpoints
    
    def _extract_models(self, content: str) -> list:
        """Extract model/schema/entity names from file content."""
        models = set()
        for pattern in self.MODEL_PATTERNS:
            for match in pattern.finditer(content):
                name = match.group(1)
                # Filter out common false positives
                if name not in ('Base', 'Model', 'Schema', 'Object', 'String', 'Number', 'Boolean'):
                    models.add(name)
        return sorted(models)
    
    def _detect_tech_stack(self, root_dir: str) -> dict:
        """Detect the technology stack from package files."""
        stack = {"backend": [], "frontend": [], "database": []}
        
        # Check package.json
        pkg_path = os.path.join(root_dir, "package.json")
        if os.path.isfile(pkg_path):
            try:
                import json
                with open(pkg_path, 'r', encoding='utf-8') as f:
                    pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                
                if "express" in deps: stack["backend"].append("Express.js")
                if "fastify" in deps: stack["backend"].append("Fastify")
                if "koa" in deps: stack["backend"].append("Koa")
                if "react" in deps: stack["frontend"].append("React")
                if "vue" in deps: stack["frontend"].append("Vue")
                if "next" in deps: stack["frontend"].append("Next.js")
                if "mongoose" in deps: stack["database"].append("MongoDB/Mongoose")
                if "sequelize" in deps: stack["database"].append("SQL/Sequelize")
                if "prisma" in deps or "@prisma/client" in deps: stack["database"].append("Prisma")
                if "pg" in deps: stack["database"].append("PostgreSQL")
                if "mysql2" in deps: stack["database"].append("MySQL")
            except Exception:
                pass
        
        # Check requirements.txt (Python)
        req_path = os.path.join(root_dir, "requirements.txt")
        if os.path.isfile(req_path):
            try:
                with open(req_path, 'r', encoding='utf-8') as f:
                    reqs = f.read().lower()
                if "flask" in reqs: stack["backend"].append("Flask")
                if "fastapi" in reqs: stack["backend"].append("FastAPI")
                if "django" in reqs: stack["backend"].append("Django")
                if "sqlalchemy" in reqs: stack["database"].append("SQLAlchemy")
                if "pymongo" in reqs: stack["database"].append("MongoDB/PyMongo")
                if "psycopg" in reqs: stack["database"].append("PostgreSQL")
            except Exception:
                pass
        
        # Check pom.xml / build.gradle (Java)
        if os.path.isfile(os.path.join(root_dir, "pom.xml")):
            stack["backend"].append("Spring/Java")
        if os.path.isfile(os.path.join(root_dir, "build.gradle")):
            stack["backend"].append("Gradle/Java")
        
        return stack
    
    def gather_context(self, root_dir=None, target_path=None):
        """
        Smart context gathering — reads project files with prioritization,
        extracts API endpoints, detects tech stack, and generates project tree.
        
        Args:
            root_dir: Root directory to scan (defaults to current working directory)
            target_path: Optional specific file or directory to focus on
        """
        if root_dir is None:
            root_dir = os.getcwd()
        
        # If target_path specified, resolve it
        if target_path:
            # Normalize Desktop/Documents prefix to absolute path
            import re
            prefix_match = re.match(r'^(Desktop|Documents)[/\\](.+)$', target_path, re.IGNORECASE)
            if prefix_match:
                folder_name = prefix_match.group(1)
                rest = prefix_match.group(2)
                target = str(Path.home() / folder_name / rest)
            elif os.path.isabs(target_path):
                target = target_path
            else:
                target = os.path.join(root_dir, target_path)
            
            # If still not found, try resolving from Desktop/Documents
            if not os.path.exists(target):
                for base in [Path.home() / "Desktop", Path.home() / "Documents", Path.cwd()]:
                    candidate = base / target_path.strip('/\\')
                    if candidate.exists():
                        target = str(candidate)
                        break
                else:
                    return f"Error: Project or file not found at '{target_path}'. Please check the path and try again."
            
            if os.path.isfile(target):
                # Single file context — just return it
                try:
                    with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if len(content) > Config.MAX_FILE_SIZE:
                        content = content[:Config.MAX_FILE_SIZE] + "\n...[TRUNCATED]..."
                    return f"\n--- FILE: {target} ---\n{content}\n"
                except Exception as e:
                    return f"Error reading {target}: {e}"
            elif os.path.isdir(target):
                root_dir = target
        
        # ===== BUILD CONTEXT =====
        
        context_parts = []
        all_endpoints = []
        all_models = set()
        
        # 1. Project structure tree
        tree = self._build_project_tree(root_dir)
        context_parts.append(f"=== PROJECT STRUCTURE ===\n{tree}\n")
        
        # 2. Detect tech stack
        stack = self._detect_tech_stack(root_dir)
        stack_lines = []
        for category, techs in stack.items():
            if techs:
                stack_lines.append(f"  {category.title()}: {', '.join(techs)}")
        if stack_lines:
            context_parts.append(f"=== TECH STACK ===\n" + "\n".join(stack_lines) + "\n")
        
        # 3. Collect and prioritize files
        all_files = []
        ignore_dirs = Config.IGNORE_DIRS
        ignore_exts = Config.IGNORE_EXTS
        code_exts = Config.CODE_EXTS
        
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                _, ext = os.path.splitext(file)
                ext_lower = ext.lower()
                
                if ext_lower in ignore_exts:
                    continue
                if code_exts and ext_lower not in code_exts:
                    continue
                
                file_path = os.path.join(root, file)
                priority = self._get_file_priority(file)
                all_files.append((priority, file_path))
        
        # Sort by priority (lower number = higher priority)
        all_files.sort(key=lambda x: x[0])
        
        logger.info(f"[CodeContext] Found {len(all_files)} code files in {root_dir}, reading by priority...")
        
        # 4. Read files in priority order
        total_size = 0
        file_count = 0
        file_contexts = []
        
        for priority, file_path in all_files:
            if total_size >= Config.MAX_TOTAL_CONTEXT_SIZE:
                file_contexts.append("\n...[CONTEXT LIMIT REACHED]...")
                break
            if file_count >= Config.MAX_CONTEXT_FILES:
                file_contexts.append(f"\n...[FILE LIMIT REACHED — {Config.MAX_CONTEXT_FILES} files max]...")
                break
            
            try:
                file_size = os.path.getsize(file_path)
                if file_size > Config.MAX_FILE_SIZE * 2:
                    rel_path = os.path.relpath(file_path, root_dir)
                    file_contexts.append(f"\n--- FILE: {rel_path} [SKIPPED — {file_size} bytes, too large] ---")
                    file_count += 1
                    continue
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract API endpoints and models
                endpoints = self._extract_api_endpoints(file_path, content)
                all_endpoints.extend(endpoints)
                
                models = self._extract_models(content)
                all_models.update(models)
                
                # Truncate large files
                if len(content) > Config.MAX_FILE_SIZE:
                    content = content[:Config.MAX_FILE_SIZE] + "\n...[TRUNCATED]..."
                
                rel_path = os.path.relpath(file_path, root_dir)
                priority_label = {1: "★ HIGH", 2: "● MEDIUM", 3: "◆ STANDARD"}.get(priority, "")
                header = f"\n--- FILE: {rel_path} {priority_label} ---\n"
                file_context = header + content + "\n"
                file_contexts.append(file_context)
                total_size += len(file_context)
                file_count += 1
                
            except Exception as e:
                logger.debug(f"Skipped unreadable file {file_path}: {e}")
        
        # 5. Build API summary
        if all_endpoints or all_models:
            api_summary = "\n=== API SUMMARY ===\n"
            if all_endpoints:
                api_summary += "Routes found:\n" + "\n".join(all_endpoints) + "\n"
            if all_models:
                api_summary += f"Models/Schemas: {', '.join(sorted(all_models))}\n"
            context_parts.append(api_summary)
        
        # 6. Combine: tree + stack + API summary + file contents
        context_parts.append(f"\n=== SOURCE FILES ({file_count} files) ===")
        context_parts.extend(file_contexts)
        
        full_context = "\n".join(context_parts)
        
        if not file_contexts:
            return "No code files found in the specified directory."
        
        logger.info(f"[CodeContext] Gathered {file_count} files, {len(all_endpoints)} endpoints, "
                     f"{len(all_models)} models, {total_size} chars total")
        
        return full_context

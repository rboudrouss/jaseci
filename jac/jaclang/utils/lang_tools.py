"""Language tools for the Jaclang project."""

import ast as py_ast
import inspect
import json
import os
import sys
from typing import Any, List, Optional, Type

import jaclang.compiler.unitree as uni
from jaclang.compiler.passes.main import PyastBuildPass
from jaclang.compiler.passes.main.cfg_build_pass import cfg_dot_from_file
from jaclang.compiler.passes.tool.doc_ir_gen_pass import DocIRGenPass
from jaclang.compiler.program import JacProgram
from jaclang.compiler.unitree import UniScopeNode
from jaclang.runtimelib.utils import read_file_with_encoding
from jaclang.utils.helpers import auto_generate_refs, pascal_to_snake


class AstKidInfo:
    """Information about a kid."""

    def __init__(self, name: str, typ: str, default: Optional[str] = None) -> None:
        """Initialize."""
        self.name = name
        self.typ = typ
        self.default = default


class UniNodeInfo:
    """Meta data about AST nodes."""

    type_map: dict[str, type] = {}

    def __init__(self, cls: type) -> None:
        """Initialize."""
        self.cls = cls
        self.process(cls)

    def process(self, cls: Type[uni.UniNode]) -> None:
        """Process UniNode class."""
        self.name = cls.__name__
        self.doc = cls.__doc__
        UniNodeInfo.type_map[self.name] = cls
        self.class_name_snake = pascal_to_snake(cls.__name__)
        self.init_sig = inspect.signature(cls.__init__)
        self.kids: list[AstKidInfo] = []
        for param_name, param in self.init_sig.parameters.items():
            if param_name not in [
                "self",
                "parent",
                "kid",
                "line",
                "sym_tab",
            ]:
                param_type = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else "Any"
                )
                param_default = (
                    param.default if param.default != inspect.Parameter.empty else None
                )
                self.kids.append(AstKidInfo(param_name, param_type, param_default))


class AstTool:
    """Ast tools."""

    def __init__(self) -> None:
        """Initialize."""
        module = sys.modules[uni.__name__]
        source_code = inspect.getsource(module)
        classes = inspect.getmembers(module, inspect.isclass)
        uni_node_classes = [
            UniNodeInfo(cls)
            for _, cls in classes
            if issubclass(cls, uni.UniNode)
            and cls.__name__
            not in [
                "UniNode",
                "UniScopeNode",
                "UniCFGNode",
                "ClientFacingNode",
                "ProgramModule",
                "OOPAccessNode",
                "WalkerStmtOnlyNode",
                "Source",
                "EmptyToken",
                "AstSymbolNode",
                "AstSymbolStubNode",
                "AstAccessNode",
                "Literal",
                "AstDocNode",
                "AstImplNeedingNode",
                "AstSemStrNode",
                "PythonModuleAst",
                "AstAsyncNode",
                "AstElseBodyNode",
                "AstTypedVarNode",
                "AstImplOnlyNode",
                "Expr",
                "AtomExpr",
                "ElementStmt",
                "ArchBlockStmt",
                "EnumBlockStmt",
                "CodeBlockStmt",
                "NameAtom",
                "ArchSpec",
                "MatchPattern",
            ]
        ]

        self.ast_classes = sorted(
            uni_node_classes,
            key=lambda cls: source_code.find(f"class {cls.name}"),
        )

    def pass_template(self) -> str:
        """Generate pass template."""
        output = (
            "import jaclang.compiler.unitree as ast\n"
            "from jaclang.compiler.passes import Pass\n\n"
            "class SomePass(Pass):\n"
        )

        def emit(to_append: str) -> None:
            """Emit to output."""
            nonlocal output
            output += "\n    " + to_append

        for cls in self.ast_classes:
            emit(
                f"def exit_{cls.class_name_snake}(self, node: ast.{cls.name}) -> None:"
            )
            emit('    """Sub objects.\n')

            for kid in cls.kids:
                emit(
                    f"    {kid.name}: {kid.typ}{' =' + str(kid.default) if kid.default else ''},"
                )

            emit('    """\n')
        output = (
            output.replace("jaclang.compiler.unitree.", "")
            .replace("typing.", "")
            .replace("<enum '", "")
            .replace("'>", "")
            .replace("<class '", "")
            .replace("ForwardRef('", "")
            .replace("')", "")
        )
        return output

    def py_uni_nodes(self) -> str:
        """List python ast nodes."""
        from jaclang.compiler.passes.main import PyastBuildPass

        visit_methods = [
            method for method in dir(py_ast._Unparser) if method.startswith("visit_")  # type: ignore
        ]
        node_names = [method.replace("visit_", "") for method in visit_methods]
        pass_func_names = []
        for name, value in inspect.getmembers(PyastBuildPass):
            if name.startswith("proc_") and inspect.isfunction(value):
                pass_func_names.append(name.replace("proc_", ""))
        output = ""
        missing = []
        for i in node_names:
            nd = pascal_to_snake(i)
            this_func = (
                f"def proc_{nd}(self, node: py_ast.{i}) -> ast.UniNode:\n"
                + '    """Process python node."""\n\n'
            )
            if nd not in pass_func_names:
                missing.append(this_func)
            output += this_func
        for i in missing:
            output += f"# missing: \n{i}\n"
        return output

    def ir(self, args: List[str]) -> str:
        """Generate a AST, SymbolTable tree for .jac file, or Python AST for .py file."""
        error = (
            "Usage: ir <choose one of (sym / sym. / ast / ast. / docir / "
            "pyast / py / unparse / esast / es / semantic / semantic.)> <.py or .jac file_path>"
        )
        if len(args) != 2:
            return error

        output, file_name = args
        prog = JacProgram()
        if not os.path.isfile(file_name):
            return f"Error: {file_name} not found"

        if file_name.endswith((".jac", ".py")):
            [base, mod] = os.path.split(file_name)
            base = base if base else "./"

            if file_name.endswith(".py"):
                file_source = read_file_with_encoding(file_name)
                parsed_ast = py_ast.parse(file_source)
                if output == "pyast":
                    return f"\n{py_ast.dump(parsed_ast, indent=2)}"
                try:
                    rep = PyastBuildPass(
                        ir_in=uni.PythonModuleAst(
                            parsed_ast,
                            orig_src=uni.Source(file_source, file_name),
                        ),
                        prog=prog,
                    ).ir_out
                    print(rep.unparse())

                    ir = prog.compile(
                        use_str=rep.unparse(),
                        file_path=file_name[:-3] + ".jac",
                        no_cgen=True,
                    )
                except Exception as e:
                    return f"Error While Jac to Py AST conversion: {e}"
            else:
                # For semantic registry extraction, we need type checking to populate usages
                if output in ["semantic", "semantic."]:
                    ir = prog.compile(file_name, no_cgen=True, type_check=True)
                else:
                    ir = prog.compile(file_name, no_cgen=True)

            match output:
                case "sym":
                    out = ""
                    for module_ in prog.mod.hub.values():
                        mod_name = module_.name
                        t = "#" * len(mod_name)
                        out += f"##{t}##\n# {mod_name} #\n##{t}##\n{module_.sym_tab.sym_pp()}\n"
                    return out
                case "sym.":
                    return (
                        ir.sym_tab.sym_printgraph()
                        if isinstance(ir.sym_tab, UniScopeNode)
                        else "Sym_tab is None."
                    )
                case "ast":
                    out = ""
                    for module_ in prog.mod.hub.values():
                        mod_name = module_.name
                        t = "#" * len(mod_name)
                        out += f"##{t}##\n# {mod_name} #\n##{t}##\n{module_.pp()}\n"
                    return out
                case "ast.":
                    return ir.printgraph()
                case "cfg.":
                    return cfg_dot_from_file(file_name)
                case "unparse":
                    return ir.unparse()
                case "pyast":
                    ir = prog.compile(file_name)
                    return (
                        f"\n{py_ast.dump(ir.gen.py_ast[0], indent=2)}"
                        if isinstance(ir.gen.py_ast[0], py_ast.AST)
                        else "Compile failed."
                    )
                case "docir":
                    return str(DocIRGenPass(ir, prog).ir_out.gen.doc_ir)
                case "py":
                    ir = prog.compile(file_name)
                    return (
                        f"\n{ir.gen.py}"
                        if isinstance(ir.gen.py[0], str)
                        else "Compile failed."
                    )
                case "esast":
                    from jaclang.compiler.passes.ecmascript import (
                        EsastGenPass,
                        es_node_to_dict,
                    )
                    import json

                    esast_pass = EsastGenPass(ir, prog)
                    es_ir = esast_pass.ir_out
                    if es_ir.gen.es_ast:
                        return f"\n{json.dumps(es_node_to_dict(es_ir.gen.es_ast), indent=2)}"
                    else:
                        return "ECMAScript AST generation failed."
                case "es":
                    from jaclang.compiler.passes.ecmascript import EsastGenPass
                    from jaclang.compiler.passes.ecmascript.es_unparse import es_to_js

                    esast_pass = EsastGenPass(ir, prog)
                    es_ir = esast_pass.ir_out
                    if es_ir.gen.es_ast:
                        return f"\n{es_to_js(es_ir.gen.es_ast)}"
                    else:
                        return "ECMAScript code generation failed."
                case "semantic":
                    # JSON output of semantic registry
                    return self._serialize_semantic_registry_json(prog, file_name)
                case "semantic.":
                    # DOT graph output of semantic registry
                    return self._serialize_semantic_registry_dot(prog, file_name)
                case _:
                    return f"Invalid key: {error}"
        else:
            return "Not a .jac or .py file, or invalid command for file type."

    def autodoc_uninode(self) -> str:
        """Generate mermaid markdown doc for uninodes."""
        output = ""
        for cls in self.ast_classes:
            if not len(cls.kids):
                continue
            output += f"## {cls.name}\n"
            output += "```mermaid\nflowchart LR\n"
            for kid in cls.kids:
                if "_end" in kid.name:
                    kid.name = kid.name.replace("_end", "_end_")
                typ_str = str(kid.typ)
                arrow = "-.->" if "Optional" in typ_str else "-->"
                typ = (
                    typ_str.replace("typing.", "")
                    .replace("jaclang.compiler.unitree.", "")
                    .replace("Optional[", "")
                    .replace("Union[", "")
                    .replace("SubTag[", "")
                    .replace("Sequence[", "")
                    .replace("list[", "list - ")
                    .replace("]", "")
                    .replace("|", ",")
                )
                output += f"{cls.name} {arrow}|{typ}| {kid.name}\n"
            output += "```\n\n"
            output += f"{cls.doc} \n\n"
        return output

    def automate_ref(self) -> str:
        """Automate the reference guide generation."""
        return auto_generate_refs()

    def gen_parser(self) -> str:
        """Generate static parser."""
        from jaclang.compiler import generate_static_parser

        generate_static_parser(force=True)
        return "Parser generated."

    def semantic(self, args: List[str]) -> str:
        """Extract and serialize the Semantic Registry for a .jac file or module.

        The Semantic Registry is the symbol table with semantic annotations attached.
        It's built after AST generation and symbol table construction, including all
        semantic strings defined via 'sem' statements.

        Args:
            args: List containing output format and file path
                  Format can be 'semantic' for JSON or 'semantic.' for DOT graph

        Returns:
            JSON or DOT representation of the semantic registry
        """
        error = (
            "Usage: semantic <.jac file_path> or semantic. <.jac file_path>\n"
            "  semantic  - Output semantic registry as JSON\n"
            "  semantic. - Output semantic registry as DOT graph"
        )
        if len(args) != 2:
            return error

        output, file_name = args
        prog = JacProgram()

        if not os.path.isfile(file_name):
            return f"Error: {file_name} not found"

        if not file_name.endswith(".jac"):
            return "Error: Only .jac files are supported for semantic registry extraction"

        try:
            # Compile up to the point where semantic registry is complete
            # This runs: Parse -> SymTabBuild -> DeclImplMatch -> SemanticAnalysis -> SemDefMatch -> TypeCheck
            # Type checking is needed to populate symbol usages
            ir = prog.compile(file_name, no_cgen=True, type_check=True)

            if ir.has_syntax_errors:
                return "Error: File has syntax errors. Cannot extract semantic registry."

            match output:
                case "semantic":
                    # JSON output
                    return self._serialize_semantic_registry_json(prog, file_name)
                case "semantic.":
                    # DOT graph output
                    return self._serialize_semantic_registry_dot(prog, file_name)
                case _:
                    return f"Invalid output format: {output}\n{error}"

        except Exception as e:
            return f"Error while extracting semantic registry: {e}"

    def _serialize_semantic_registry_json(
        self, prog: JacProgram, target_file: str
    ) -> str:
        """Serialize the semantic registry to JSON format.

        Args:
            prog: The JacProgram containing compiled modules
            target_file: The target file path to filter symbols

        Returns:
            JSON string representation of the semantic registry
        """
        registry: dict[str, Any] = {
            "version": "1.0",
            "modules": {},
        }

        # Normalize target file path for comparison
        import os

        target_file_abs = os.path.abspath(target_file)

        # Process all modules in the program hub
        for mod_path, module in sorted(prog.mod.hub.items()):
            module_data = self._extract_module_symbols(
                module.sym_tab, module.name, target_file_abs
            )
            # Only include modules that have symbols from the target file
            if module_data["symbols"] or module_data["sub_scopes"]:
                registry["modules"][mod_path] = module_data

        # Use json.dumps with sort_keys for deterministic output
        return json.dumps(registry, indent=2, sort_keys=True)

    def _extract_module_symbols(
        self, scope: UniScopeNode, scope_name: str, target_file: str
    ) -> dict[str, Any]:
        """Recursively extract symbols from a scope.

        Args:
            scope: The symbol table scope to extract from
            scope_name: Name of the current scope
            target_file: The target file path to filter symbols

        Returns:
            Dictionary containing scope information and symbols
        """
        import os

        scope_data: dict[str, Any] = {
            "scope_name": scope_name,
            "scope_type": scope.__class__.__name__,
            "symbols": {},
            "sub_scopes": {},
        }

        # Extract symbols in this scope
        for sym_name, symbol in sorted(scope.names_in_scope.items()):
            # Skip built-in symbols
            if sym_name in ["builtins", "__builtins__"]:
                continue

            # Check if this symbol is defined in the target file
            is_in_target_file = False
            if symbol.defn:
                for defn in symbol.defn:
                    if os.path.abspath(defn.loc.mod_path) == target_file:
                        is_in_target_file = True
                        break

            # Check if this symbol is used in the target file
            has_usage_in_target = False
            if symbol.uses:
                for use in symbol.uses:
                    if os.path.abspath(use.loc.mod_path) == target_file:
                        has_usage_in_target = True
                        break

            # Skip symbols that are neither defined nor used in the target file
            if not is_in_target_file and not has_usage_in_target:
                continue

            symbol_data: dict[str, Any] = {
                "name": symbol.sym_name,
                "type": str(symbol.sym_type),
                "access": str(symbol.access),
                "dotted_name": symbol.sym_dotted_name,
                "imported": symbol.imported,
            }

            # Add semantic string if present
            if symbol.semstr:
                symbol_data["semantic"] = symbol.semstr

            # Add definitions only from the target file
            if symbol.defn:
                target_defns = [
                    {
                        "file": defn.loc.mod_path,
                        "line": defn.loc.first_line,
                        "column": defn.loc.col_start,
                        "end_line": defn.loc.last_line,
                        "end_column": defn.loc.col_end,
                    }
                    for defn in symbol.defn
                    if os.path.abspath(defn.loc.mod_path) == target_file
                ]
                if target_defns:
                    symbol_data["definitions"] = target_defns
                elif not is_in_target_file:
                    # This is a builtin/external symbol used in target file
                    # Just indicate it's external
                    symbol_data["external"] = True

            # Add usages only from the target file
            if symbol.uses:
                target_uses = [
                    {
                        "file": use.loc.mod_path,
                        "line": use.loc.first_line,
                        "column": use.loc.col_start,
                        "end_line": use.loc.last_line,
                        "end_column": use.loc.col_end,
                    }
                    for use in symbol.uses
                    if os.path.abspath(use.loc.mod_path) == target_file
                ]
                if target_uses:
                    symbol_data["usages"] = target_uses

            scope_data["symbols"][sym_name] = symbol_data

        # Recursively process child scopes
        for kid_scope in scope.kid_scope:
            if kid_scope.scope_name == "builtins":
                continue
            kid_data = self._extract_module_symbols(
                kid_scope, kid_scope.scope_name, target_file
            )
            # Only include child scopes that have relevant symbols
            if kid_data["symbols"] or kid_data["sub_scopes"]:
                scope_data["sub_scopes"][kid_scope.scope_name] = kid_data

        return scope_data

    def _serialize_semantic_registry_dot(
        self, prog: JacProgram, target_file: str
    ) -> str:
        """Serialize the semantic registry to DOT graph format.

        Args:
            prog: The JacProgram containing compiled modules
            target_file: The target file path to filter symbols

        Returns:
            DOT graph string representation of the semantic registry
        """
        import os

        target_file_abs = os.path.abspath(target_file)
        dot_lines = ["digraph semantic_registry {"]
        dot_lines.append('  rankdir=TB;')
        dot_lines.append('  node [shape=box, style=rounded];')
        dot_lines.append("")

        node_id_counter = [0]  # Use list to allow mutation in nested function
        node_ids: dict[int, int] = {}

        def get_node_id(obj: Any) -> int:
            """Get or create a unique node ID."""
            obj_id = id(obj)
            if obj_id not in node_ids:
                node_ids[obj_id] = node_id_counter[0]
                node_id_counter[0] += 1
            return node_ids[obj_id]

        def escape_label(text: str) -> str:
            """Escape special characters for DOT labels."""
            return text.replace('"', '\\"').replace("\n", "\\n")

        def add_scope_to_graph(
            scope: UniScopeNode, parent_id: Optional[int] = None
        ) -> None:
            """Recursively add scope and symbols to DOT graph."""
            scope_id = get_node_id(scope)
            scope_label = f"{scope.__class__.__name__}\\n{escape_label(scope.scope_name)}"

            # Add scope node
            dot_lines.append(
                f'  n{scope_id} [label="{scope_label}", fillcolor=lightblue, style="rounded,filled"];'
            )

            # Connect to parent if exists
            if parent_id is not None:
                dot_lines.append(f"  n{parent_id} -> n{scope_id};")

            # Add symbols in this scope
            for sym_name, symbol in sorted(scope.names_in_scope.items()):
                if sym_name in ["builtins", "__builtins__"]:
                    continue

                # Check if this symbol is relevant to the target file
                is_in_target_file = False
                has_usage_in_target = False

                if symbol.defn:
                    for defn in symbol.defn:
                        if os.path.abspath(defn.loc.mod_path) == target_file_abs:
                            is_in_target_file = True
                            break

                if symbol.uses:
                    for use in symbol.uses:
                        if os.path.abspath(use.loc.mod_path) == target_file_abs:
                            has_usage_in_target = True
                            break

                # Skip symbols not relevant to target file
                if not is_in_target_file and not has_usage_in_target:
                    continue

                sym_id = get_node_id(symbol)
                sym_label = f"{escape_label(sym_name)}\\n{symbol.sym_type}"

                # Mark external symbols
                if not is_in_target_file:
                    sym_label += "\\n(external)"

                # Add semantic string to label if present
                if symbol.semstr:
                    # Truncate long semantic strings
                    sem_preview = (
                        symbol.semstr[:50] + "..."
                        if len(symbol.semstr) > 50
                        else symbol.semstr
                    )
                    sym_label += f'\\n"{escape_label(sem_preview)}"'

                # Color based on whether semantic string exists and if external
                if not is_in_target_file:
                    fillcolor = "lightgray"  # External symbols
                elif symbol.semstr:
                    fillcolor = "lightgreen"  # Has semantic string
                else:
                    fillcolor = "white"  # Local symbol without semantic

                dot_lines.append(
                    f'  n{sym_id} [label="{sym_label}", fillcolor={fillcolor}, style="rounded,filled"];'
                )
                dot_lines.append(f"  n{scope_id} -> n{sym_id};")

            # Recursively process child scopes
            for kid_scope in scope.kid_scope:
                if kid_scope.scope_name == "builtins":
                    continue
                add_scope_to_graph(kid_scope, scope_id)

        # Process all modules
        for mod_path, module in sorted(prog.mod.hub.items()):
            add_scope_to_graph(module.sym_tab)

        dot_lines.append("}")
        return "\n".join(dot_lines)

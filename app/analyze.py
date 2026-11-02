import os, ast, json
files = [
    'setup_planner.py', 'execution_readiness.py', 'entry_confirmation.py',
    'direction_confidence.py', 'tp_path_validator.py', 'liquidity.py',
    'imbalance.py', 'crash_regime.py', 'crash_continuation_setup.py',
    'range_sweep_model.py', 'smc_context.py', 'trend_regime.py',
    'trend_direction.py', 'wave_template_detector.py', 'mtf_elliott_confirmation.py'
]
path = 'c:/Users/User/Desktop/wicky_ai_v2/elliott-wave-backend/app/models'

output = []
for f in files:
    full = os.path.join(path, f)
    if os.path.exists(full):
        with open(full, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = ast.parse(content)
            funcs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    doc = ast.get_docstring(node)
                    doc = doc.strip().split('\n')[0] if doc else "No docstring"
                    funcs.append(f"{node.name}: {doc}")
            output.append(f"--- {f} ---")
            output.extend(funcs)

with open('c:/Users/User/Desktop/wicky_ai_v2/elliott-wave-backend/app/func_summary.txt', 'w', encoding='utf-8') as out:
    out.write("\n".join(output))
print("Done writing to func_summary.txt")

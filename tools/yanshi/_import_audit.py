import ast
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python')
PKG = os.path.join(ROOT, 'infernux')


def module_exists(name: str) -> bool:
    parts = name.split('.')
    path = os.path.join(ROOT, *parts)
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, '__init__.py'))
    return os.path.isfile(path + '.py')


def main(paths):
    missing = {}
    for path in paths:
        tree = ast.parse(open(path, encoding='utf-8').read(), path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.append(node.module)
                for alias in node.names:
                    names.append(node.module + '.' + alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name)
            for name in names:
                if not name.startswith('infernux'):
                    continue
                # attribute imports resolve to the parent module; only flag modules
                if not module_exists(name):
                    base = name.rsplit('.', 1)[0]
                    if module_exists(base):
                        continue
                    missing.setdefault(name, []).append(path)
    for name in sorted(missing):
        print('MISSING', name)
        for path in sorted(set(missing[name]))[:3]:
            print('   ', os.path.relpath(path, ROOT))
    print('missing modules:', len(missing))


if __name__ == '__main__':
    main(sys.argv[1:])

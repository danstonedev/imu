import sys, os, traceback
print('PY=', sys.version)
print('CWD=', os.getcwd())
print('sys.path[0]=', sys.path[0])
print('sys.executable=', sys.executable)
print('sys.path=')
for p in sys.path:
    print('  -', p)
try:
    import importlib.util as _ilu
    spec = _ilu.find_spec('app')
    print('find_spec("app") ->', spec)
except Exception as _e:
    print('find_spec error:', _e)

print('\nDir listing of CWD:')
for name in os.listdir('.'):
    print('  *', repr(name))

print('\nmeta_path finders:')
for f in sys.meta_path:
    print('  -', f)

app_path = os.path.join(os.getcwd(), 'app.py')
print('\nDirect load attempt via SourceFileLoader from', app_path)
try:
    import importlib.machinery as _ilm
    import types as _types
    if os.path.exists(app_path):
        loader = _ilm.SourceFileLoader('app_direct', app_path)
        mod = _types.ModuleType('app_direct')
        loader.exec_module(mod)
        print('Direct load OK, module has attributes:', list(vars(mod).keys())[:10])
    else:
        print('app.py not found on disk')
except Exception as e:
    print('Direct load FAIL:', type(e).__name__, e)

def try_import(name):
    print(f'\n--- importing {name} ---')
    try:
        mod = __import__(name, fromlist=['*'])
        print(f'OK: {name} -> {mod}')
        return mod
    except Exception as e:
        print(f'FAIL: {name}: {e.__class__.__name__}: {e}')
        traceback.print_exc()

try_import('app')
try_import('core')
try_import('core.pipeline.pipeline')

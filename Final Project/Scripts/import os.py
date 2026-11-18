import os

path = r"C:\Users\A-FF-L1-D25\Documents\Final Project\Nat2023us"

for root, dirs, files in os.walk(path):
    # Skip hidden and __pycache__
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    level = root.replace(path, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for f in sorted(files):
        size = os.path.getsize(os.path.join(root, f))
        if size > 1_000_000:
            s = f'{size/1_000_000:.1f} MB'
        else:
            s = f'{size/1_000:.0f} KB'
        print(f'{indent}  {f}  ({s})')
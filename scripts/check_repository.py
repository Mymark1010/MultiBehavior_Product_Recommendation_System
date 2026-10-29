"""Check active project files for obsolete task paths, broken links and invalid notebooks."""
import ast
import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]
# These strings are regression guards only, never dataset fields or objectives.
RETIRED=[r'mua\s+(?:trong\s+)?7\s*ngày',r'seven[- ]day',r'7[- ]day purchase',
         r'purchase_targets',r'label_horizon_days',r'has_purchase_target',
         r'queries_with_purchase',r'incomplete_label_window',r'build_user_features',
         r'prepare_multibehavior',r'configs/multibehavior\.json',
         r'01_prepare_retailrocket',r'data/multibehavior',r'reports/multibehavior']


def main():
    errors=[]
    count=0
    for path in ROOT.rglob('*'):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(x in rel.parts for x in ('.git','.venv','.tools','__pycache__')):
            continue
        if 'data' in rel.parts and rel != Path('data/README.md'):
            continue
        if path.resolve()==Path(__file__).resolve():
            continue
        if path.suffix in ('.docx','.pptx'):
            with ZipFile(path) as z:
                text='\n'.join(' '.join(ElementTree.fromstring(z.read(n)).itertext())
                               for n in z.namelist() if n.endswith('.xml'))
        elif path.suffix in ('.md','.py','.json','.yml','.yaml','.ipynb','.txt'):
            text=path.read_text(encoding='utf-8')
        else:
            continue
        count+=1
        for pattern in RETIRED:
            if re.search(pattern,text,re.IGNORECASE):
                errors.append(f'{rel}: obsolete task/path ({pattern})')
        if path.suffix=='.py':
            ast.parse(text,filename=str(rel))
        if path.suffix=='.json':
            json.loads(text)
        if path.suffix=='.ipynb':
            notebook=json.loads(text)
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']),filename=f'{rel}:cell{i}')
                if any(x.get('output_type')=='error' for x in cell.get('outputs',[])):
                    errors.append(f'{rel}: saved notebook error in cell {i}')
        if path.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                target=target.strip('<>').split('#')[0]
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):
                    continue
                if not (path.parent/target).exists():
                    errors.append(f'{rel}: broken local link ({target})')
    config=json.loads((ROOT/'configs/nextstep.json').read_text())
    if config.get('task')!='next_step':
        errors.append('The canonical configuration must select next_step')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Checked {count} active text/Office files: no obsolete task paths, broken local links or saved notebook errors.')


if __name__=='__main__':
    main()

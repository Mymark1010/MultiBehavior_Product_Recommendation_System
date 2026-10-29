"""Execute the repository's read-only notebooks using this Python environment.

Requires requirements-notebooks.txt and an already prepared full dataset.
"""
import json
import os
import sys
from pathlib import Path


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    root=Path(__file__).resolve().parents[1]
    scratch=root/'.tools/jupyter'
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ['IPYTHONDIR']=str(scratch/'ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(scratch/'runtime')
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
    completed=[]
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv[0]=sys.executable
        client=NotebookClient(notebook,km=manager,timeout=180,
                              resources={'metadata':{'path':str(root/'notebooks')}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook,path)
        completed.append(path.name)
        print('Executed:',path.name,flush=True)
    (root/'reports/nextstep/notebook_validation.json').write_text(
        json.dumps({'python':sys.version.split()[0],'notebooks_executed':completed,
                    'all_cells_completed_without_error':True},indent=2)+'\n')


if __name__=='__main__':
    main()

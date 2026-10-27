from pathlib import Path
import os,subprocess
root=Path(__file__).resolve().parents[1];prefix=root.parents[1]/'.redun/dataset-utility/utility/environments/audiolidar'
env=dict(os.environ,LD_LIBRARY_PATH=str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),PYTHONPATH=str(root/'snapshots/generator'),MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
subprocess.run([str(prefix/'bin/python'),'-B','-u','scripts/generate_common_menu.py','--rank','0','--limit','1'],cwd=root,env=env,check=True)

"""Push an authorized branch without persisting credentials in Git config."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repository, remote, branch = sys.argv[1:]
assert remote in ('https://github.com/yyttt6/TileOPs.git',
                  'https://github.com/yyttt6/TileOPs.github.io.git')
with tempfile.NamedTemporaryFile(mode='w',prefix='tileops-askpass-',delete=False) as file:
    file.write('''#!/usr/bin/python3
from pathlib import Path
import sys
print('x-access-token' if 'Username' in sys.argv[1] else Path('/tmp/tileops-remote-ci-auth').read_text().strip())
''')
    helper = Path(file.name)
helper.chmod(0o700)
try:
    env = dict(os.environ,GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git','-C',repository,'-c','credential.helper=','push',remote,
                    'HEAD:refs/heads/'+branch],env=env,check=True)
finally:
    helper.unlink(missing_ok=True)

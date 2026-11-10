"""Standard site convenience functions for original CMake's isolated Python probe."""
import runpy
import site
import sys

site.setquit()
args = sys.argv[1:]
if args[0] == '-c':
    sys.argv = ['-c', *args[2:]]
    exec(compile(args[1], '<command>', 'exec'), {'__name__': '__main__'})
else:
    sys.argv = args
    runpy.run_path(args[0], run_name='__main__')

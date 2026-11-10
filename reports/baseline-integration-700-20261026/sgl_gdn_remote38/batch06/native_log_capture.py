"""CPU-only capture of CANN attribution events before automatic log rotation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading


class NativeLogCapture:
    def __init__(self, directory, output):
        self.directory, self.output = Path(directory), Path(output)
        self.stop = threading.Event()
        self.state = {}
        for path in self.directory.rglob('*.log'):
            st=path.stat(); self.state[(str(path),st.st_dev,st.st_ino)]=(st.st_size,0)
        self.events = 0
        self.error = None
        self.markers = tuple(s.encode() for s in (
            'Register Kernel succ', 'bin path is ', 'StreamLaunchKernel',
            'LogTaskDetails', 'GetJsonValueFromJsonFile', 'single op compile success.'))

    def __enter__(self):
        self.stream = self.output.open('x')
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()
        return self

    def _scan(self):
        progress = False
        for path in self.directory.rglob('*.log'):
            try:
                stat = path.stat()
                key = str(path), stat.st_dev, stat.st_ino
                offset, number = self.state.get(key, (0, 0))
                if stat.st_size < offset:
                    offset, number = 0, 0
                if stat.st_size == offset:
                    continue
                with path.open('rb') as source:
                    source.seek(offset)
                    chunk = source.read(4 * 1024 * 1024)
                complete = chunk.split(b'\n')
                tail = complete.pop()
                offset += len(chunk) - len(tail)
                for line in complete:
                    number += 1
                    if any(marker in line for marker in self.markers):
                        self.stream.write(json.dumps({
                            'file': str(path.resolve()), 'line': number,
                            'text': line.decode(errors='replace'),
                            'captured_at': datetime.now(timezone.utc).isoformat(),
                        }) + '\n')
                        self.events += 1
                self.state[key] = offset, number
                progress |= bool(complete)
            except FileNotFoundError:
                # Rotated between stat/open: retain all events already captured.
                continue
        self.stream.flush()
        return progress

    def _run(self):
        try:
            while not self.stop.is_set():
                self._scan()
                self.stop.wait(0.1)
            while self._scan():
                pass
        except Exception as error:
            self.error = repr(error)

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self.stream.close()
        self.output.with_suffix('.summary.json').write_text(json.dumps({
            'events': self.events, 'error': self.error,
            'cpu_only': True, 'source_directory': str(self.directory.resolve()),
            'sha256': hashlib.sha256(self.output.read_bytes()).hexdigest(),
        }, indent=2) + '\n')

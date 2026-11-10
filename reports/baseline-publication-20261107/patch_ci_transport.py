from pathlib import Path
import json
P=Path(__file__).resolve().parent; f=P/'ci/.github/workflows/ascend950.yml';s=f.read_text();pin=json.loads((P/'site-published.json').read_text())['commit']
sources='''  sources:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ inputs.revision || github.sha }}
          fetch-depth: 0
          persist-credentials: false
      - uses: actions/checkout@v4
        with:
          ref: ascend950
          path: ci-harness
          persist-credentials: false
      - uses: actions/checkout@v4
        with:
          repository: yyttt6/TileOPs.github.io
          ref: SITE_PIN
          path: baseline-cache
          persist-credentials: false
      - name: Package exact sources for isolated runners
        run: |
          mkdir -p "$RUNNER_TEMP/ci-sources"
          git rev-parse HEAD > "$RUNNER_TEMP/ci-sources/source.commit"
          git archive --format=tar HEAD > "$RUNNER_TEMP/ci-sources/source.tar"
          tar -rf "$RUNNER_TEMP/ci-sources/source.tar" .git
          gzip "$RUNNER_TEMP/ci-sources/source.tar"
          git -C ci-harness archive --format=tar.gz HEAD > "$RUNNER_TEMP/ci-sources/harness.tar.gz"
          git -C baseline-cache archive --format=tar.gz HEAD data/baseline-cache > "$RUNNER_TEMP/ci-sources/baseline.tar.gz"
          cd "$RUNNER_TEMP/ci-sources"
          sha256sum source.commit source.tar.gz harness.tar.gz baseline.tar.gz > SHA256SUMS
      - uses: actions/upload-artifact@v4
        with:
          name: ascend950-sources
          path: ${{ runner.temp }}/ci-sources
          overwrite: true
          if-no-files-found: error
          retention-days: 7
'''.replace('SITE_PIN',pin)
extract='''      - uses: actions/download-artifact@v4
        with:
          name: ascend950-sources
          path: ${{ runner.temp }}/ci-archives-${{ github.run_id }}-${{ github.run_attempt }}-${{ github.job }}
      - name: Unpack verified sources
        env:
          ARCHIVES: ${{ runner.temp }}/ci-archives-${{ github.run_id }}-${{ github.run_attempt }}-${{ github.job }}
          DESTINATION: ${{ runner.temp }}/ci-source-${{ github.run_id }}-${{ github.run_attempt }}-${{ github.job }}
        run: |
          cd "$ARCHIVES"
          sha256sum --check SHA256SUMS
          mkdir -p "$DESTINATION"
          tar --no-same-owner -xzf source.tar.gz -C "$DESTINATION"
          test "$(git -C "$DESTINATION" rev-parse HEAD)" = "$(cat source.commit)"
          test -z "$(git -C "$DESTINATION" status --porcelain)"
'''
wd='        working-directory: ${{ runner.temp }}/ci-source-${{ github.run_id }}-${{ github.run_attempt }}-${{ github.job }}\n'
# Modify only the existing runner jobs, preserving their gates and tests.
parts=s.split('jobs:\n',1);jobs=parts[1]
for name in ['preflight','pr-performance','device']:
 start=jobs.index('  '+name+':\n');nextpos=jobs.find('\n  ',start+len('  '+name+':\n'))
 # Job boundaries have exactly two spaces followed by a non-space.
 import re
 m=re.search(r'\n  [a-z][a-z-]*:\n',jobs[start+1:]);end=start+1+m.start()+1 if m else len(jobs)
 block=jobs[start:end]
 a=block.index('      - uses: actions/checkout@v4');b=block.index('      - name:',a)
 rest=block[b:].replace('        run:',wd+'        run:').replace('$GITHUB_WORKSPACE','$PWD')
 unpack=extract
 if name=='pr-performance':unpack+='''          mkdir -p "$DESTINATION/ci-harness" "$DESTINATION/baseline-cache"
          tar --no-same-owner -xzf harness.tar.gz -C "$DESTINATION/ci-harness"
          tar --no-same-owner -xzf baseline.tar.gz -C "$DESTINATION/baseline-cache"
'''
 block=block[:a]+unpack+rest
 if name=='preflight':block=block.replace('  preflight:\n','  preflight:\n    needs: sources\n')
 jobs=jobs[:start]+block+jobs[end:]
f.write_text(parts[0]+'jobs:\n'+sources+jobs)

"""Private environment path bootstrap; invoked entirely within host guard."""
import hashlib,json,os,runpy,sys
from pathlib import Path
package=os.environ.get("ASCEND_CLASSIC_FLA_PACKAGE")
if package:
    sys.path.insert(0,str(Path(package).resolve().parent))
args=sys.argv[1:]
out=Path(args[args.index("--out")+1])
entry=Path(__file__).resolve()
receipt=out/"launch_environment.json"
receipt.write_text(json.dumps({"launcher":str(entry),"launcher_sha256":hashlib.sha256(entry.read_bytes()).hexdigest(),"package":package,"package_parent_in_sys_path":str(Path(package).resolve().parent) in sys.path if package else None}))
os.environ["BASELINE_CACHE_LAUNCH_RECEIPT"]=str(receipt)
sys.argv=args
runpy.run_path(args[0],run_name="__main__")

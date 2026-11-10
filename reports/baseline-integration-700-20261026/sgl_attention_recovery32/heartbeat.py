from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent;now=datetime.datetime.now(datetime.timezone.utc).isoformat();data=json.loads((P/'RESULT.json').read_text());attempts=sorted((P/'attempts').iterdir(),key=lambda f:f.stat().st_mtime);a=attempts[-1]
(P/'HEARTBEAT.json').write_text(json.dumps({'utc':now,'updated_at':now,'worker':'sgl_attention_recovery32','status':'running','current_attempt':a.name,'guard_finished':(a/'GUARD_RESULT.json').exists(),'counts':data['counts'],'immutable_interim':'DELIVERY_INTERIM.json'},indent=2))

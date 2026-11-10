"""Read-only host prerequisites; this entry never imports torch, ACL or TileLang."""

import argparse
import json
import socket
from pathlib import Path


def inspect(config_path):
    errors = []
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get("host_reservation_confirmed") is not True:
        errors.append("Host-shared reservation, PID visibility and device mapping are not confirmed")
    for key in ("guard", "clock_config"):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            errors.append(f"Missing audited site {key}")
    lock = Path("/run/lock/ascend-npu/device-0.lock")
    if not lock.is_file():
        errors.append("The host device reservation lock is not mounted")
    return {
        "status": "blocked" if errors else "ready",
        "hostname": socket.gethostname(),
        "device_nodes": sorted(str(p) for p in Path("/dev").glob("davinci[0-9]*")),
        "pid1": Path("/proc/1/comm").read_text().strip(),
        "hardware_started": False,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 78 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

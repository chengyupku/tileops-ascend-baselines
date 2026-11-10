import pathlib,os,json,platform,datetime
p=pathlib.Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libacl_op_compiler.so');s=p.stat()
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=platform.node(),machine=platform.machine(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stat=dict(device=s.st_dev,major=os.major(s.st_dev),minor=os.minor(s.st_dev),inode=s.st_ino,size=s.st_size),mountinfo=pathlib.Path('/proc/self/mountinfo').read_text())))

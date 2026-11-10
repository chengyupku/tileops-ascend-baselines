"""Read function symbols from host ELF and embedded device ELFs without loading them."""
import struct,pathlib,json,sys

def functions(path):
 data=pathlib.Path(path).read_bytes();out=[];start=0
 while True:
  base=data.find(b'\x7fELF',start)
  if base<0:break
  start=base+4
  try:
   if data[base+4:base+6]!=b'\x02\x01':continue
   h=struct.unpack_from('<HHIQQQIHHHHHH',data,base+16);shoff=h[5];entsz=h[10];num=h[11]
   if entsz!=64 or num>10000 or base+shoff+num*64>len(data):continue
   sec=[struct.unpack_from('<IIQQQQIIQQ',data,base+shoff+i*64) for i in range(num)]
   for s in sec:
    if s[1] not in (2,11) or s[9]!=24 or s[6]>=num:continue
    st=sec[s[6]];strings=data[base+st[4]:base+st[4]+st[5]]
    for off in range(base+s[4],base+s[4]+s[5],24):
     name,info,other,idx,val,size=struct.unpack_from('<IBBHQQ',data,off)
     if info&15!=2 or size==0 or idx==0:continue
     end=strings.find(b'\0',name);n=strings[name:end].decode(errors='replace')
     if 'mix_aic' in n or 'mix_aiv' in n or ((base>0 or h[1]==4137) and info>>4!=0):out.append({'elf_offset':base,'machine':h[1],'name':n,'size':size})
  except (struct.error,IndexError):continue
 return out
if __name__=='__main__':print(json.dumps({p:functions(p) for p in sys.argv[1:]},indent=2))

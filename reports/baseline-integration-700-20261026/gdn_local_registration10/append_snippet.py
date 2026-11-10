
from .integration700_gdn_local_registration10 import wrap as _wrap_gdn_local10
PROVIDERS['GDNFwdOp'] = _wrap_gdn_local10(PROVIDERS.get('GDNFwdOp'))

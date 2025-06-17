from .models import TimesNet
from .models import DLinear
from .models import PatchTST
from .models import MICN
from .models import iTransformer
from .models import Koopa
from .models import FreTS
from .models import TimeMixer
from .models import TiDE
from .models import Minusformer

model_dict = {
    "TimesNet": TimesNet.Model,
    "DLinear": DLinear.Model,
    "PatchTST": PatchTST.Model,
    "MICN": MICN.Model,
    "iTransformer": iTransformer.Model,
    "Koopa": Koopa.Model,
    "TiDE": TiDE.Model,
    "FreTS": FreTS.Model,
    "TimeMixer": TimeMixer.Model,
    "Minusformer": Minusformer.Model,
}

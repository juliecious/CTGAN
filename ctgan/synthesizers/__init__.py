from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan.synthesizers.tvae import TVAESynthesizer
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from ctgan.synthesizers.fl_dpctgan import FLDPCTGANSynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'DPCTGANSynthesizer',
    'FLDPCTGANSynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }

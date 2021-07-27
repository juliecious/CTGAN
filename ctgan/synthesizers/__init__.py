from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan.synthesizers.tvae import TVAESynthesizer
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from ctgan.synthesizers.adp_ctgan import ADPCTGANSynthesizer


__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'DPCTGANSynthesizer',
    'ADPCTGANSynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }

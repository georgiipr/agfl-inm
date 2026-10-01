from .. import AttentionSpec
from .config import DEFAULTS
from .layer import MultiHeadAttention

SPEC = AttentionSpec('mha', DEFAULTS, lambda options, tokens: MultiHeadAttention(
    options['dim'], options['heads']))

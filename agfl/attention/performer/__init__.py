from .. import AttentionSpec
from .config import DEFAULTS
from .layer import Performer

SPEC = AttentionSpec('performer', DEFAULTS, lambda options, tokens: Performer(options))

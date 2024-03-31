from src.base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes training options.It also includes shared options defined in BaseOptions."""
    def initialize(self, parser):
        # Initialize base options
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--input_type",  type=str, default="standard")
        return parser


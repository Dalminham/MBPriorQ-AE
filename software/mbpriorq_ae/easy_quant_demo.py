class EasyQuantDemo():
    def __init__(self, args):
        self.name         = args["name"]
        self.quant_device = args["device"]
        self.quant_bit    = args["quant_bit"]
        self.quant_sym    = args["quant_sym"]
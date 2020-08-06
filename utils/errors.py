class EnvError(Exception):

    def __init__(self, *args):
        self.args = args

class NotInitiateError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'environment is uninitialized, use reset to initialize it befor running.'
    
    def __str__(self):
        return self.errorinfo

class EnvTerminatedError(EnvError):

    def __init__(self):
        super().__init__(self)
        self.errorinfo = 'environment is terminated, please reset it befor running.'
    
    def __str__(self):
        return self.errorinfo
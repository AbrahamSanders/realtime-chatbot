class Identity:
    def __init__(self):
        self.name = "unknown"
        self.age = "unknown"
        self.sex = "unknown"

    @staticmethod
    def default_identities():
        return {
            "S1": Identity(),
            "S2": Identity()
        }
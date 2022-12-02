class Identity:
    def __init__(self, name="unknown", age="unknown", sex="unknown"):
        self.name = name
        self.age = age
        self.sex = sex

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @staticmethod
    def default_identities():
        return {
            "S1": Identity(),
            "S2": Identity()
        }
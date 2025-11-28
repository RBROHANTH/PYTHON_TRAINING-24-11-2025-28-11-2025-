class Animal:
    @abstractmethod
    def sound(self):
        pass

    @abstractmethod
    def Dog(self):
        pass
class Dog(Animal):
    def sound(self):
        return "Woof Woof"
    
    def Dog(self):
        return "This is a Dog class implementing Animal interface"
    


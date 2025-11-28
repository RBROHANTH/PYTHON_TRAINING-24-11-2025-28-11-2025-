class Histroy:
    def __init__(self):
        self.stack = []

    def visit(self,site):
        '''
        this function will give you the last visited site

        param site: site name

        return: None
        '''
        print("I have Visited:", site)
        self.stack.append(site)

    def back(self):
        if len(self.stack) > 1:
            removed = self.stack.pop()
            print("I am back from:", removed)
            print("Now I am at:", self.stack[-1])
        else:
            print("No history is here")

    def show(self):
        print("Browsing History:", self.stack)

data = Histroy()

data.visit("google.com")
data.visit("youtube.com")
data.visit("facebook.com")

data.show()
data.back()
data.show()
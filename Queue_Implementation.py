class CallCenter:
    def __init__(self):
        self.queue = []

    def new_call(self,customer):
        print(customer,"joined the call")
        self.queue.append(customer)

    def attend(self):
        if len(self.queue) > 0:
            customer = self.queue.pop(0)
            print("Attending the call of : ",customer)

        else:
            print("No calls in the queue")
    
    def show(self):
        print("Current Call Queue:", self.queue)


call = CallCenter()

call.new_call("Vivekananda")
call.new_call("god")
call.new_call("Yamrajan")
call.attend()
call.show()
call.attend()
call.show()
call.attend()
call.show()
class Employee:
    def __init__(self, eid, name, post, city, state, pin, bs, ta, hra):
        self.eid = eid
        self.name = name
        self.post = post
        self.city = city
        self.state = state
        self.pin = pin
        self.bs = bs
        self.ta = int(ta)
        self.hra = hra
    def show_info(self):
        print(f"Employee ID: {self.eid}")
        print(f"Employee Name: {self.name}")
    def show_address(self):
        print(f"City: {self.city}")
        print(f"State: {self.state}")
        print(f"Pin: {self.pin}")
    def show_salary(self):
        print(f"Basic Salary: {self.bs}")
        print(f"TA: {self.ta}")
        print(f"HRA: {self.hra}")

emp1 = Employee(101, "Rohanth R B", "Game Developer", "Thaliapatty,Velianai,Karur", "Tamil Nadu", "639 188", 70000, "rohanth", hra=8000)

emp2 = Employee(102, "Santhosh V", "OS Engineer", "Perundurai,Erode", "Tamil Nadu", "638 060", 65000, 4500, 7000)
print("--------------------------------------------------")
print("--------------------------------------------------")
emp1.show_info()
emp1.show_address() 
emp1.show_salary()
print("--------------------------------------------------")
print("--------------------------------------------------")
emp2.show_info()
emp2.show_address()
emp2.show_salary()
print("--------------------------------------------------")
print("--------------------------------------------------")



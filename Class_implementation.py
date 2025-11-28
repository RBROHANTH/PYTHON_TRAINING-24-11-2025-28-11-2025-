class A:
    def method1(self):
        return "Method 1 from Class A"

    def method2(self):
        return "Method 2 from Class A"
    
class B:
    def method3(self):
        return "Method 3 from Class B" 
    def method4(self):
        return "Method 4 from Class B"
    
class C(A,B):
    def method5(self):
        return "Method 5 from Class C"

ob = B()
oa = A()
oc = C()

print(oc.method1())
print(oc.method2())
print(oc.method3())
print(oc.method4())
print(oc.method5())
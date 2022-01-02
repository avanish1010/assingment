#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Create a class called Account which will be an abstract class for three other classes called 
CheckingAccount, SavingsAccount and BusinessAccount. 
Manage credits and debits from these accounts through an ATM style program."""


# In[30]:


from abc import ABC,abstractmethod

class Account(ABC):
    @abstractmethod
    def login(self):
        pass
    @abstractmethod
    def singup(self):
        pass
    @abstractmethod
    def deposit(self):
        pass
    @abstractmethod
    def withdrawl(self):
        pass
    
    


# In[31]:


class Checking_Account(Account):
    
    def __init__(self,balance=0,user_info={},account_number=1000):
        self.balance=balance
        self.user_info=user_info
        self.account_number=account_number
        
    def singup(self):
        name=input("Enter your name: ")
        self.account_number+=1
        print("Your account number is: ",self.account_number)
        age=int(input("Enter your age: "))
        address=input("Enter your address: ")
        pin=input("Enter your pin number and the length of the pin number must be four digits: ")
        if age>=18 and len(pin)==4:
            self.user_info['name']=name
            self.user_info['account_number']=self.account_number
            self.user_info['age']=age
            self.user_info['address']=address
            self.user_info['PIN']=pin
            print("Your details are",self.user_info)
            return "singup successfull"
        else:
            return "Not eligible to create a bank account"
        
    def login(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        if self.user_info['account_number']==self.account_number:
            print("loign successfull")
            return "login successfull"
        else:
            return "Inputed wrong account number "
            
    def deposit(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to deposit: ")
            new_balance= int(input())
            self.balance+=new_balance
            print("You deposited:", new_balance)
            return "Your total amount in the account is: ", self.balance
        else:
            return "Please check your account number or your pin"
    
    def withdrawl(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to withdraw: ")
            your_withdrawl= int(input())
            if self.balance>= your_withdrawl:
                if your_withdrawl<=5000:
                    bank_charge=(your_withdrawl*2)/100
                    your_withdrawl+=bank_charge
                    self.balance-=your_withdrawl
                    print("You withdrarwl: ", your_withdrawl)
                    return "Your remaning balance", self.balance
                else:
                    return "Withdrawl amount exceeded"
            else:
                return "You do not have sufficient balance to withdraw"
        else:
            return "Please check your account number or your pin"


# In[32]:


user1=Checking_Account()
user1.singup()
user1.login()
user1.deposit()
user1.withdrawl()


# In[ ]:


class Savings_account(Account):
    def __init__(self,balance=0,user_info={},account_number=1000):
        self.balance=balance
        self.user_info=user_info
        self.account_number=account_number
        
    def singup(self):
        name=input("Enter your name: ")
        self.account_number+=1
        print("Your account number is: ",self.account_number)
        age=int(input("Enter your age: "))
        address=input("Enter your address: ")
        pin=input("Enter your pin number and the length of the pin number must be four digits: ")
        if age>=18 and len(pin)==4:
            self.user_info['name']=name
            self.user_info['account_number']=self.account_number
            self.user_info['age']=age
            self.user_info['address']=address
            self.user_info['PIN']=pin
            print("Your details are",self.user_info)
            return "singup successfull"
        else:
            return "Not eligible to create a bank account"
        
    def login(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        if self.user_info['account_number']==self.account_number:
            print("Login successfull")
            return "login successfull"
        else:
            return "Inputed wrong account number "
            
    def deposit(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to deposit: ")
            new_balance= int(input())
            self.balance+=new_balance
            print("You deposited:", new_balance)
            return "Your total amount in the account is: ", self.balance
        else:
            return "Please check your account number or your pin"
    
    def withdrawl(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to withdraw: ")
            your_withdrawl= int(input())
            if self.balance>= your_withdrawl:
                if your_withdrawl<=5000:
                    self.balance-=your_withdrawl
                    print("You withdrarwl: ", your_withdrawl)
                    return "Your remaning balance", self.balance
                else:
                    return "Withdrawl amount exceeded"
            else:
                return "You do not have sufficient balance to withdraw"
        else:
            return "Please check your account number or your pin"


# In[40]:


user=Savings_account()
user.singup()
user.login()
user.deposit()
user.withdrawl()


# In[28]:


class Business_Account(Account):
    def __init__(self,balance=0,user_info={},account_number=1000):
        self.balance=balance
        self.user_info=user_info
        self.account_number=account_number
        
    def singup(self):
        name=input("Enter your name: ")
        self.account_number+=1
        print("Your account number is: ",self.account_number)
        age=int(input("Enter your age: "))
        address=input("Enter your address: ")
        pin=input("Enter your pin number and the length of the pin number must be four digits: ")
        if age>=18 and len(pin)==4:
            self.user_info['name']=name
            self.user_info['account_number']=self.account_number
            self.user_info['age']=age
            self.user_info['address']=address
            self.user_info['PIN']=pin
            print("Your details are",self.user_info)
            return "singup successfull"
        else:
            return "Not eligible to create a bank account"
        
    def login(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        if self.user_info['account_number']==self.account_number:
            print("login successfull")
            return "login successfull"
        else:
            return "Inputed wrong account number "
            
    def deposit(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to deposit: ")
            new_balance= int(input())
            self.balance+=new_balance
            print("You deposited:", new_balance)
            return "Your total amount in the account is: ", self.balance
        else:
            return "Please check your account number or your pin"
    
    def withdrawl(self):
        print("please enter your bank account number")
        your_account_number= int(input())
        print("please enter your bank PIN number")
        your_pin_number= input()
        if self.user_info['account_number']==your_account_number and self.user_info['PIN']==your_pin_number: 
            print("Enter the amount you want to withdraw: ")
            your_withdrawl= int(input())
            if self.balance>= your_withdrawl:
                if your_withdrawl<=5000:
                    bank_charge=(your_withdrawl*6)/100
                    your_withdrawl+=bank_charge
                    self.balance-=your_withdrawl
                    print("You withdrarwl: ", your_withdrawl)
                    return "Your remaning balance", self.balance
                else:
                    return "Withdrawl amount exceeded"
            else:
                return "You do not have sufficient balance to withdraw"
        else:
            return "Please check your account number or your pin"


# In[29]:


user2=Business_Account()
user2.singup()
user2.login()
user2.deposit()
user2.withdrawl()


# In[ ]:





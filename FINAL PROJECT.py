#!/usr/bin/env python
# coding: utf-8

# In[8]:


class MENU():
    def __init__(self,foodlist,new_id):
        self.foodlist=[]
        self.new_id=new_id
    def menu_add(self,name,price,quantity,discount,stock, unit):
        self.foodlist.append({'name':name,'id':self.new_id,
                              'price':price,'quantity':quantity,
                              'discount':discount,'stock':stock, 
                             'unit':unit})
        self.new_id=self.new_id+1
        return "YOUR FOODLIST IS :   ",self.foodlist
    
    def menu_remove(self,new_id):
        for i in range(len(self.foodlist)):
            if self.foodlist[i]['id'] == new_id:
                del self.foodlist[i]
                break
    
    def menu_edit(self,new_id,key, new_val):
        for a in range(len(self.foodlist)):
            if self.foodlist[a]['id']== new_id:
                self.foodlist[a][key]= new_val
    def display_menu(self):
        for i in range(len(self.foodlist)):
            print  ("YOUR MENU IS:  item_name : {}  \n id:{}  \n  price:{}  \n quantity:{}  \n discount:{}   \n stock: {}".format(self.foodlist[i]['name'] ,self.foodlist[i]['id'],self.foodlist[i]['price'], self.foodlist[i]['quantity'],self.foodlist[i]['discount'],self.foodlist[i]['stock']))     
      
    def order_method(self,new_id):
        for a in range(len(self.foodlist)):
            if self.foodlist[a]['id'] == new_id:
                order=self.foodlist[a]
                if self.foodlist[a]['stock'] >= self.foodlist[a]['quantity']:
                    self.foodlist[a]['stock']=self.foodlist[a]['stock']-self.foodlist[a]['quantity']
                    return order
                else:
                    print("OUT OF STOCK")
                    return False
            else:
                continue
        else:
            return "PLEASE CHECK ID"


# In[9]:


class ADMIN():
    def __init__(self, username, pw):
        self.username=username
        self.password=pw
        self.is_admin=False
        
        
    def Admin(self,username,password):
        if username.startswith('admin'):
            self.is_admin=True
            return 'ADMIN LOGIN SUCCESSFUL'
        else:
            return "PLEASE CHECK THE USERNAME"
        
    def add_food(self,menu_object,name,price,quantity,discount,stock, unit):
        if self.is_admin==True:
            return menu_object.menu_add(name,price,quantity,discount,stock, unit)
        else:
            return  "ACCESS DENIED"
        
    def remove_food(self,menu, new_id):
        if self.is_admin==True:
            return menu.menu_remove(new_id)
        else:
            return  "ACCESS DENIED"
    
    def edit_food(self,menu, new_id,key,new_val):
        if self.is_admin==True:
            return menu.menu_edit(new_id,key,new_val)
        else:
            return  "ACCESS DENIEDS"


# In[10]:


class USER():
    def __init__(self,fullname,phone_number,address,email_address,username,password,orders):
        self.fullname=fullname
        self.phone_number=phone_number
        self.address=address
        self.email_address=email_address
        self.username=username
        self.password=password
        self.is_user=False
        self.orders=orders
        self.user_data=[]
        
        
    def user_details(self):
        self.user_data.append({ 'fullname':self.fullname,
                    'phone_number':self.phone_number,
                    'address':self.address,
                    'email_address':self.email_address,
                    'username':self.username,
                    'password':self.password})
        return "YOUR DETAILS ARE:  ",self.user_data
    
    
    def login(self,username,password):
        if  self.username.startswith('user'):
            self.is_user=True
            return 'USER ACCESS GRANTED !!!'
        else:
            return "PLEASE CHECK THE USERNAME"
        
        
    def select_an_option(self):
        user_choice= "Select among the three options:  Option 1->Place New Order, Option 2->Order History, Option 3-> Update Profile"
        return "YOUR OPTIONS ARE : ",user_choice
    
    
    def place_order (self,menu_object,new_id):
        if not self.is_user:
            print("ACCESS DENIED")
            return False
        else:
            print(menu_object.display_menu())  
            order = menu_object.order_method(new_id)
            if not order :
                print('Could not place')
                return False
            else: 
                self.orders.append(order)
                return "YOUR ORDER IS  :", order
    def previous_orders(self):
            return "YOUR PREVIOUS ORDERS ARE: ", self.orders
    def edit_profile(self,key,value):
        for i in range(len(self.user_data)):
            self.user_data[i][key]=value
            print ("PROFILE UPDATED SUCCESSFULLY")
            return self.user_data


# In[14]:


menu=MENU([],1000)
person1=ADMIN('adminAvanish','avanish10')
person1.Admin('adminavanish','avanish10')
person1.add_food(menu,'pasta',450 ,1,50,100, 'units')
person1.add_food(menu,'baby corn',300 ,1,40,87, 'units')
person1.add_food(menu, 'truffle cake',450, 500,40,1, 'kgs')
user=USER('a','b','c','d','usere','f',[])
user.login('useravanish','avanish10')


# In[15]:


if __name__ == '__main__':
    print("Select among the three options:  Option 1->Place New Order, Option 2->Order History, Option 3-> Update Profile")
    user_choice_input=int(input())
    if user_choice_input==1:
        print(user.place_order(menu,1001))
    elif user_choice_input==2:
        print(user.previous_orders())
    elif user_choice_input==3:
        print(user.user_details())
        print(user.edit_profile(input("ENTER THE KEY:\n  "),input("ENTER THE VALUE:\n  ")))
        
    else:
        print("wrong input")


# In[ ]:





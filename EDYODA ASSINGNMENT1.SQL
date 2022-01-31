USE EDYODA;

Create Table SalesPeople
(Snum  INT Primary key,
Sname VARCHAR(30) Unique ,
City Varchar(30), Comm INT);

INSERT INTO SalesPeople VALUES 
(1001,'Peel','London',12),
(1002 ,'Serres','Sanjose',13),
(1004,'Motika','London',11),
(1007,'Rifkin','Barcelona',15),
(1003,'Axelrod','Newyork',10);

Create Table Customers
(Cnum INT Primary Key,
Cname VARCHAR(30),
City Varchar(30) not null,
Snum INT,
foreign key (Snum) references SalesPeople(Snum));

INSERT INTO Customers VALUES
(2001,'Hoffman','London',1001),
(2002,'Giovanni','Rome',1003),
(2003,'Liu','Sanjose',1002),
(2004,'Grass','Berlin',1002),
(2006,'Clemens','London',1001),
(2008,'Cisneros','Sanjose',1007),
(2007,'Pereira','Rome',1004);

CREATE TABLE Orders
(Onum INT Primary key,
Amt DEcimal,
Odate DAte,
Cnum INT,
Snum INT,
foreign key (Cnum) references  Customers(Cnum),
foreign key (Snum) references SalesPeople (Snum));

INSERT INTO Orders VALUES 
(3001,18.69,'1990-10-03',2008,1007),
(3003,767.19,'1990-10-03',2001,1001),
(3002,1900.10,'1990-10-03',2007,1004),
(3005,5160.45,'1990-10-03',2003,1002),
(3006,1098.16,'1990-10-03',2008,1007),
(3009,1713.23,'1990-10-04',2002,1003),
(3007,75.75,'1990-10-04',2004,1002),
(3008,4273.00,'1990-10-05',2006,1001),
(3010,1309.95,'1990-10-06',2004,1002),
(3011,9891.88,'1990-10-06',2006,1001);


--Count the number of Salesperson whose name begin with ‘a’/’A’.
Select * FRom SalesPeople WHERE Sname LIKE 'A%';

--Display all the Salesperson whose all orders worth is more than Rs. 2000.
Select Snum,SUm(Amt) FRom Orders GRoup by Snum;
select SalesPeople.Sname,orders.Amt
from Orders
inner join SalesPeople on orders.Snum = SalesPeople.Snum
where orders.Amt > 2000;

--Count the number of Salesperson belonging to Newyork.
SElect count(*) from Salespeople where city='Newyork';

--Display the number of Salespeople belonging to London and belonging to Paris.
SElect count(*) from Salespeople where city='London' or city='paris';

--Display the number of orders taken by each Salesperson and their date of orders.
select SalesPeople.Sname, count(*), orders.Odate from SalesPeople
inner join Orders on SalesPeople.Snum = orders.Snum
group by SalesPeople.Sname,orders.Odate;

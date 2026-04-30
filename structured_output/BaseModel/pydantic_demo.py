from pydantic import BaseModel, Field
import json

class Person(BaseModel):
    name : str
    age : int
    isalive : bool


# Normal way of creating an instance of the Person class
candidate = Person(name="John Doe", age=30, isalive=True) # Pydantic will convert the integer 1 to a boolean True and 0 to False. However, it's generally better to provide the correct data type for clarity and to avoid potential issues in the future.


class Employee(BaseModel):
    name : str
    age : int
    email : str 
    salary : float = Field(..., description="The salary of the employee in USD") # Using Field to add a description for the salary field. The ... indicates that this field is required.

# The Dictionary way of creating an instance of the Employee class .
employee_dict = {
    "name" : "Jane Smith",
    "age" : 25,
    "email" : "jane.smith@example.com",
    "salary" : 75000.00
}
employee = Employee(**employee_dict) # Unpacking the dictionary to create an instance of Employee

# Printing the instances to verify that they were created correctly.
print(candidate)
print(employee)


print()

# Converting the Person instance to a dictionary using the dict() method provided by Pydantic. This will give us a dictionary representation of the Person instance, which we can then use to access individual fields or manipulate as needed.
Person_dict = dict(candidate) # Converting the Person instance back to a dictionary
print(type(Person_dict)) 
print(Person_dict)
print(Person_dict['name']) # Accessing the name field from the dictionary representation of the Person instance.


Person_json = candidate.model_dump_json() # This method will return a JSON string representation of the Person instance. It's a convenient way to serialize the data for storage or transmission.
print(Person_json)
print(type(Person_json)) 

# print(Person_json['name']) # This will raise an error because Person_json is a string, not a dictionary. To access the name field, we need to parse the JSON string back into a dictionary first.

print()
Person_json_dict = json.loads(Person_json) # Parsing the JSON string back into a dictionary
print(type(Person_json_dict))
print(Person_json_dict)
print(Person_json_dict['name']) # Now we can access the name field from the dictionary representation of the Person instance.
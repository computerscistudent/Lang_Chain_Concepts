from typing import TypedDict

class person(TypedDict):
    Name : list[str]
    age : int
    isalive : bool

cnadidate : person = {
    'Name' : ["Abhi","himanshi","ishan","shiv"],
    'age' : 22,
    'isalive' : True
}

print(cnadidate)
import pdb

a_list = [[i] * i for i in range(1, 6)]

print(a_list)

while a_list:

    a_element = a_list.pop(0)
    
    for a in a_element:
        print(a)

print(a_list)

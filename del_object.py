class DelObj:

    def __init__(self, num):
        
        self.num = num


if __name__ == '__main__':

    obj_list = []
    for i in range(10):
        obj_list.append(DelObj(i))
        
    for i, obj in enumerate(obj_list):
        if not obj.num % 2:
            del obj_list[i]
        
    for obj in obj_list:
        print(obj.num)






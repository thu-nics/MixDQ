def func(a,b,c,*args,**kwargs):
    print(a,b,c)
    for arg in args:
        print('arg:',arg)
    for kwarg in kwargs:
        print('kwarg',kwarg)

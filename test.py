my_dict = {8:'u',4:'t',9:'z',10:'j',5:'k',3:'s'}

new_filt = dict(
    filter(
        lambda val: val[0] % 3 == 0,
        my_dict.items()
    )
)

print("Filter dictionary:",new_filt)


a = (1,2,3)
print(list(a))
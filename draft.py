import numpy as np

a = ('a', 'b', 'c', 1)

def fitr(tuple, words, first_characters): 
    
    arr = np.array(tuple[0][:-1])
    if len(words) != arr.shape[0]: return False
    else:
        return np.all(arr == np.array(words)) and tuple[0][-1].startswith(first_characters)


array = np.array([a])
print(type(array[0][:-1]))

b = ('the', 'great', 'dog')
print(b[:-9])

print(array[0][0])
print(np.array(('a', 'b')))

a = [1, 2, 3]
print(tuple(a))

a = ('h',)
print(a[:-1])
# '''map(): This function applies a given function to each element of an iterable and returns 
# a map object,which can be converted into a list, tuple, or set. '''
# numbers = [1, 2, 3, 4, 5]
# squared_numbers = list(map(lambda x: x**2, numbers))
# print(squared_numbers)  # Output: [1, 4, 9, 16, 25]

# '''filter(): This function filters an iterable based on a given function that returns a 
# Boolean value. It returns a filter object, which can be converted into a list, tuple, or set.'''
# numbers = [1, 2, 3, 4, 5]
# even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
# print(even_numbers)  # Output: [2, 4]

# '''reduce(): This function applies a given function to the first two elements of an iterable, 
# then to the result and the next element, and so on, until there's only one element left. 
# It returns a single value'''
# from functools import reduce
# numbers = [1, 2, 3, 4, 5]
# product = reduce(lambda x, y: x*y, numbers)
# print(product)  # Output: 120

# '''sorted(): This function takes an iterable and returns a new sorted list based on a given 
# key function. It can also take additional arguments to control the sorting behavior.'''
# words = ['banana', 'apple', 'cherry', 'durian']
# sorted_words = sorted(words, key=lambda x: len(x))
# print(sorted_words)  # Output: ['apple', 'banana', 'cherry', 'durian']

# '''any() and all(): These functions take an iterable and return True if at least one or all 
# elements, respectively, evaluate to True based on a given function'''
# numbers = [1, 2, 3, 4, 5]
# has_even_numbers = any(map(lambda x: x % 2 == 0, numbers))
# are_all_numbers_positive = all(map(lambda x: x > 0, numbers))
# print(has_even_numbers)  # Output: True
# print(are_all_numbers_positive)  # Output: True

# '''partial(): This function allows you to create a new function from an existing function by 
# fixing some of its arguments'''
# def multiply(x, y):
#     return x * y
# # create a new function that multiplies by 2
# double = functools.partial(multiply, 2)
# print(double(3)) # Output: 6

# '''reduce(): This function applies a function of two arguments cumulatively to the items of
#  an iterable, from left to right, so as to reduce the iterable to a single value'''

# # calculate the factorial of 5 using reduce()
# factorial = functools.reduce(lambda x, y: x * y, range(1, 6))
# print(factorial) # Output: 120

# '''cache(): This function is used to cache the results of a function call. The cached result 
# is returned when the same inputs occur again. This can save time if the function takes a long 
# time to execute'''
# import functools
# def fibonacci(n):
#     if n <= 1:
#         return n
#     else:
#         return fibonacci(n-1) + fibonacci(n-2)
# print(fibonacci(10)) # Output: 55

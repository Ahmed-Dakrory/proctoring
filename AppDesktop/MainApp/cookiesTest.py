# import browser_cookie3
# import requests
# cj = browser_cookie3.chrome(domain_name="www.onlydispatch.com")
# # r = requests.get('https://www.onlydispatch.com', cookies=cj)
# # print(r.content)
# for cookie in cj:
#    print(cookie.name, cookie.value, cookie.domain)
# # cookies = cj.csrftoken
# # print(cookies)

# class LocalStorage:

#     def __init__(self, driver) :
#         self.driver = driver

#     def __len__(self):
#         return self.driver.execute_script("return window.localStorage.length;")

#     def items(self) :
#         return self.driver.execute_script( \
#             "var ls = window.localStorage, items = {}; " \
#             "for (var i = 0, k; i < ls.length; ++i) " \
#             "  items[k = ls.key(i)] = ls.getItem(k); " \
#             "return items; ")

#     def keys(self) :
#         return self.driver.execute_script( \
#             "var ls = window.localStorage, keys = []; " \
#             "for (var i = 0; i < ls.length; ++i) " \
#             "  keys[i] = ls.key(i); " \
#             "return keys; ")

#     def get(self, key):
#         return self.driver.execute_script("return window.localStorage.getItem(arguments[0]);", key)

#     def set(self, key, value):
#         self.driver.execute_script("window.localStorage.setItem(arguments[0], arguments[1]);", key, value)

#     def has(self, key):
#         return key in self.keys()

#     def remove(self, key):
#         self.driver.execute_script("window.localStorage.removeItem(arguments[0]);", key)

#     def clear(self):
#         self.driver.execute_script("window.localStorage.clear();")

#     def __getitem__(self, key) :
#         value = self.get(key)
#         if value is None :
#           raise KeyError(key)
#         return value

#     def __setitem__(self, key, value):
#         self.set(key, value)

#     def __contains__(self, key):
#         return key in self.keys()

#     def __iter__(self):
#         return self.items().__iter__()

#     def __repr__(self):
#         return self.items().__str__()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome(executable_path=r"chromedriver.exe")



print('----------------------------------------------')
print(browser.execute_script('return localStorage.getItem("token");'))

# # get the local storage
# browser.execute_script("window.localStorage;")

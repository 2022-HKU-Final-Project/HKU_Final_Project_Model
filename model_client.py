import requests


if __name__ == '__main__':
    url1 = 'http://localhost:8001/get_info'
    data = {
        'content': "我精通C++，Java等多门编程语言，扎实掌握数据结构算法等。"}
    print(requests.get(url1, params=data).json())
from multiprocess import Process, Queue, Pool


# def f(q):
#     q.put("hello world")
#     q.put()


def func(x, a):
    return x * x + a


if __name__ == "__main__":

    def f(inp):
        return func(*inp)

    p = Pool(4)
    results = p.map_async(f, [(1, 1), (2, 3), (3, 4)])

    print(results.get())

    # q = Queue()
    # p = Process(target=f, args=[q])
    # p.start()
    # print(q.get())
    # p.join()

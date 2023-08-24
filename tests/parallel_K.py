import multiprocessing

def do_func(x):
    print(x)
    return {
        'val1': x, 
        'val2': x**2
        }

def process_numbers(numbers):
    num_cpu = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cpu) as pool:
        results = pool.map(do_func, numbers)
    return results

if __name__ == '__main__':
    numbers = [i*0.5 for i in range(10)]
    results = process_numbers(numbers)
    print(results)
    print()
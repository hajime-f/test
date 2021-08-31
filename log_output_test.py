from datetime import date

def output_log(log_str):
    
    file_name = date.today().strftime('%Y-%m-%d') + '.log'
    with open(file_name, mode='a') as f:
        f.write(log_str)

if __name__ == '__main__':
    
    log_test = [True, False, False]

    test_str = ', '.join([('True' if i else 'False') for i in log_test]) + '\n'
    
    output_log(test_str)
    


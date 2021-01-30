import os


def merge_to_one(file_path):
    dir = os.path.dirname(file_path)
    new_file_path = os.path.join(dir, 'analysis_copy.json')
    my_file = open(new_file_path, "w")
    with open(file_path, 'r') as f:
        for line in f:
            if '}{' in line:
                my_file.write(',')
            else:
                my_file.write(line)
    my_file.close()


if __name__ == "__main__":
    path = "/Users/zber/ProgramDev/exp_pyTorch/results/Standard_MNIST_LeNet_20200630-003814/Standard_MNIST_LeNet__analysis_recode.json"


    merge_to_one(path)
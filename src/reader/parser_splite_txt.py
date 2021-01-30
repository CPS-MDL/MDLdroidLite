import os
# structure10 = "C:\\Users\\Zber\\Documents\\Dev_program\\FastGrownTest\\results\\standard_2_5_50_20200603-151533\\standard_2_5_50__json_in_out.json"
# structure = "C:\\Users\\Zber\\Documents\\Dev_program\\FastGrownTest\\results\\standard_6_15_150_20200603-171112\\standard_6_15_150__json_in_out.json"
# structure = "C:\\Users\\Zber\\Documents\\Dev_program\\FastGrownTest\\results\\standard_12_30_300_20200603-224535\\standard_12_30_300__json_in_out.json"
# structure = "C:\\Users\\Zber\\Documents\\Dev_program\\FastGrownTest\\results\\standard_20_50_500_20200604-013122\\standard_20_50_500__json_in_out.json"
structure = ""

dir = os.path.dirname(structure)
file_path = os.path.join(dir,'j1.json')
my_file = open(file_path, "w")

index=1
with open(structure, 'r') as f:
    for line in f:
        if line.startswith('}{'):
            my_file.write('}\n')
            my_file.close()
            index += 1
            file_path = os.path.join(dir, 'j{}.json'.format(index))
            my_file = open(file_path, "w")
            my_file.write('{\n')
        else:
            my_file.write(line)
    my_file.close()
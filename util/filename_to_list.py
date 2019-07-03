import os

f = open("results.txt", "a+")

for file in os.listdir("/home/p19g2/data_race_add/train/trainA"):
    print("Processing: " + file)
    
    file_split = file.split('_', 1)

    classification = file_split[0]
    original_name = file_split[1]

    print("F:%s C:%s O:%s", file, classification, original_name)

    f.write(original_name + " " + classification)

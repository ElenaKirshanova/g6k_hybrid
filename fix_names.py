# fixes old naming convension used up to commit 81c0b6672bf7433256cc004501e5ab2268ba1c9d 
import os, re

out_path = "./lwe_instances/reduced_lattices/"
regex = re.compile("kyb_prehybrid*")
for path, directories, files in os.walk(out_path):
    for candidate in files:
        if regex.match(candidate):
            # print(out_path+candidate)
            # command = "git mv " + out_path + candidate + " " + out_path + candidate[:-4]
            command = "mv " + out_path + candidate + " " + out_path + candidate[:-4]
            os.system( command )
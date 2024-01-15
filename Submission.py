#Aqsa Noreen
import os

def get_output_of_file(filename):
    return os.popen(f'python {filename}').read()

if __name__ == "__main__":
    descisionT = get_output_of_file("DecisionTree.py")
    print("The outputs for DecisionTree.py:")
    print(descisionT)
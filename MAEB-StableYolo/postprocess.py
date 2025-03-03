# This program receives a path to a folder where there are several files with the extension *fitness.
# Each file contains the fitness values of the individuals of a population in a generation.
# The program will find the best individual

import os
import sys
import numpy as np

def zip_best_individual(name):
        #Remove the extension of the file
        name = name[:-8]
        #Extract the folder and subfolders for the current files
        folder="/".join(name.split("/")[:-1])
        exp = "_".join(name.split("/")[-3:-1])
        name = name.split("/")[-1]
        #Get all the files that start with the name
        files = os.listdir(folder)
        files = [file for file in files if file.startswith(name)]
        #Create a zip with the files
        for file in files:
            os.system("zip " +  exp + "_" + name + ".zip " + folder + "/" + file)
        print(files)

def search_best_individual(folder):
        files = os.listdir(folder)
        best_fitness = -1
        best_individual = None
        for file in files:
                if file.endswith(".fitness"):
                    with open(folder + "/" + file) as f:
                        lines = f.readlines()
                        for line in lines:
                            fitness = float(line)
                            if fitness > best_fitness:
                                best_fitness = fitness
                                best_individual = line
                                best_file = file
        print("Best fitness: " + str(best_fitness))
        print("Best individual: " + best_individual)
        print("File name: " + folder + "/" + best_file)
        print("Prompts:")
        #The best files ends with .fitness, so we need to remove this extension
        original_best_file = best_file
        best_file = best_file[:-8]
        with open(folder + "/" + best_file + ".0.prompt.txt") as f:
            lines = f.readlines()
            print(lines)
        zip_best_individual(folder + "/" + original_best_file)
        return best_individual

def search_worst_individual(folder):
        files = os.listdir(folder)
        worst_fitness = 100
        worst_individual = None
        for file in files:
                if file.endswith(".fitness"):
                    with open(folder + "/" + file) as f:
                        lines = f.readlines()
                        for line in lines:
                            fitness = float(line)
                            if fitness < worst_fitness:
                                worst_fitness = fitness
                                worst_individual = line
                                worst_file = file
        print("Worst fitness: " + str(worst_fitness))
        print("Worst individual: " + worst_individual)
        print("File name: " + folder + "/" + worst_file)
        print("Prompts:")
        #The best files ends with .fitness, so we need to remove this extension
        worst_file = worst_file[:-8]
        with open(folder + "/" + worst_file + ".0.prompt.txt") as f:
            lines = f.readlines()
            print(lines)
        return worst_individual       

if __name__ == "__main__":
        if len(sys.argv) != 2:
                print("Usage: python postprocess.py <folder>")
                sys.exit(1)
        folder = sys.argv[1]
        best_individual = search_best_individual(folder)

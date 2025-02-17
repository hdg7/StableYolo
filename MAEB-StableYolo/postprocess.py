# This program receives a path to a folder where there are several files with the extension *fitness.
# Each file contains the fitness values of the individuals of a population in a generation.
# The program will find the best individual

import os
import sys
import numpy as np

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

def main():
        if len(sys.argv) != 2:
                print("Usage: python postprocess.py <folder>")
                sys.exit(1)
        folder = sys.argv[1]
        best_individual = search_best_individual(folder)

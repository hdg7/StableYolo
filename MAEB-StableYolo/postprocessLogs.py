# This program receives a path to a folder where there are several files with the extension *fitness.
# Each file contains the fitness values of the individuals of a population in a generation.
# The program will find the best individual

import os
import sys
import numpy as np
import re
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


classes=["banana","bear","bird","cat","dog","elephant","giraffe", "person","train","zebra"]
experiments=["MAEB","SSBSE"]

def identify_convergence(logs):
        currentConv=0
        currentAvg=0
        for log in logs:
                if (log['Average']>currentAvg):
                        currentAvg = log['Average']
                        currentConv = log['Generation']
        return currentConv,currentAvg


def identify_best(logs):
        currentConv=0
        currentAvg=0
        for log in logs:
                if (log['Best Ind']>currentAvg):
                        currentAvg = log['Best Ind']
                        currentConv = log['Generation']
        return currentConv,currentAvg

def identify_firts(logs):
        return logs[0]['Generation'],logs[0]['Average']

def identify_firts_best(logs):
        return logs[0]['Generation'],logs[0]['Best Ind']

def parse_log_line(line):
        #pattern = re.compile(r"(\d+)\s+(\d+)\s+\[\s*([\d\.]+)\]\s+\[\s*([\d\.]+)\]\s+\[\s*([\d\.]+)\]\s+\[\s*([\d\.]+)\]")
        pattern = re.compile(r"(\d+)\s+(\d+)\s+\[\s*([\d\.e-]+)\]\s+\[\s*([\d\.e-]+)\]\s+\[\s*([\d\.e-]+)\]\s+\[\s*([\d\.e-]+)\]")
            
        match = pattern.match(line)

        if match:
                return {
                        "Generation": int(match.group(1)),
                        "Evaluations": int(match.group(2)),
                        "Average": float(match.group(3)),
                        "Std": float(match.group(4)),
                        "Worst Ind": float(match.group(5)),
                        "Best Ind": float(match.group(6))
                }
        return None

def getLogs(folder):
        files = os.listdir(folder)
        logs = []
        for file in files:
                if file.startswith("output"):
                        print(file)
                        with open(folder + "/" + file) as f:
                                lines = f.readlines()
                                logs.append(lines)
        #print("".join(logs[0][-54:-3]))
        return [parse_log_line(log) for log in logs[0][-54:-3]]

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

def plot_results(cases):
        elementNames = ["banana", "bear", "bird", "cat", "dog", "elephant", "giraffe", "person", "train", "zebra"]
        results = {
                'DeepStableYolo': cases['MAEB'],
                'StableYolo': cases['SSBSE'],
                'Baseline': cases['Baseline']
        }
        # plt.figure(figsize=(12, 6))
    
        # for key, values in results.items():
        #         plt.plot(elementNames, values, marker='o', label=key)

        # plt.xlabel("Elements", fontsize=14)
        # plt.ylabel("Scores", fontsize=14)
        # plt.title("Comparison of Different Methods", fontsize=16)
        # plt.legend(fontsize=12)
        # plt.xticks(rotation=45, fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(True)
        # plt.tight_layout()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Line plot
        for key, values in results.items():
                axes[0].plot(elementNames, values, marker='o', label=key)
        axes[0].set_xlabel("Elements", fontsize=14)
        axes[0].set_ylabel("Scores", fontsize=14)
        axes[0].set_title("Comparison of Different Methods", fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].tick_params(axis='x', rotation=45, labelsize=12)
        axes[0].tick_params(axis='y', labelsize=12)
        axes[0].grid(True)
        
        # Box plot
        data = [results[key] for key in results]
        sns.boxplot(data=data, ax=axes[1])
        axes[1].set_xticklabels(results.keys(), fontsize=14)
        axes[1].set_ylabel("Scores", fontsize=14)
        axes[1].set_title("Score Distributions", fontsize=16)
        axes[1].tick_params(axis='x', labelsize=14)
        axes[1].tick_params(axis='y', labelsize=14)
        
        plt.tight_layout()
        plt.savefig("results.pdf")




        
def main():
        results={}
        resultsValues={}
        resultsZero={} 
        for elem in experiments:
                results[elem]=[]
                resultsValues[elem]=[]
                resultsZero[elem]=[]
                for name in classes:
                        print("results/" + elem + "/" + name)
                        logs=getLogs("results/" + elem + "/" + name)
                        gen,val = identify_convergence(logs)
                        results[elem].append(gen)
                        resultsValues[elem].append(val)
                        gen0,val0 = identify_firts(logs)
                        resultsZero[elem].append(val0)
        print(results)
        print(resultsValues)
        print(resultsZero)
        final={}
        final["MAEB"]=resultsValues["MAEB"]
        final["SSBSE"]=resultsValues["SSBSE"]
        final["Baseline"]=resultsZero["MAEB"]
        print("MAEB average: " + str(np.mean(resultsValues["MAEB"])))
        print("SSBSE average: " + str(np.mean(resultsValues["SSBSE"])))
        print("Baseline average: " + str(np.mean(resultsZero["MAEB"])))
        print("MAEB std: " + str(np.std(resultsValues["MAEB"])))
        print("SSBSE std: " + str(np.std(resultsValues["SSBSE"])))
        print("Baseline std: " + str(np.std(resultsZero["MAEB"])))
        print("MAEB average: " + str(np.mean(results["MAEB"])))
        print("SSBSE average: " + str(np.mean(results["SSBSE"])))
        print("MAEB std: " + str(np.std(results["MAEB"])))
        print("SSBSE std: " + str(np.std(results["SSBSE"])))
        print(scipy.stats.wilcoxon(results["MAEB"],results["SSBSE"]))
        print(scipy.stats.wilcoxon(resultsValues["MAEB"],resultsValues["SSBSE"]))
        print(scipy.stats.wilcoxon(resultsZero["MAEB"],resultsZero["SSBSE"]))
        print(scipy.stats.ttest_rel(results["MAEB"],resultsZero["MAEB"]))
        print(scipy.stats.ttest_rel(results["SSBSE"],resultsZero["SSBSE"]))
        plot_results(final)
        return final



def main2():
        results={}
        resultsValues={}
        resultsZero={} 
        for elem in experiments:
                results[elem]=[]
                resultsValues[elem]=[]
                resultsZero[elem]=[]
                for name in classes:
                        print("results/" + elem + "/" + name)
                        logs=getLogs("results/" + elem + "/" + name)
                        gen,val = identify_best(logs)
                        results[elem].append(gen)
                        resultsValues[elem].append(val)
                        gen0,val0 = identify_firts_best(logs)
                        resultsZero[elem].append(val0)
        print(results)
        print(sum(results["MAEB"]))
        print(sum(results["SSBSE"]))
        print(resultsValues)
        print(resultsZero)
        final={}
        final["MAEB"]=resultsValues["MAEB"]
        final["SSBSE"]=resultsValues["SSBSE"]
        final["Baseline"]=resultsZero["SSBSE"]
        print("MAEB average: " + str(np.mean(resultsValues["MAEB"])))
        print("SSBSE average: " + str(np.mean(resultsValues["SSBSE"])))
        print("Baseline average: " + str(np.mean(resultsZero["MAEB"])))
        print("MAEB std: " + str(np.std(resultsValues["MAEB"])))
        print("SSBSE std: " + str(np.std(resultsValues["SSBSE"])))
        print("Baseline std: " + str(np.std(resultsZero["MAEB"])))
        print(scipy.stats.wilcoxon(results["MAEB"],results["SSBSE"]))
        print(scipy.stats.wilcoxon(resultsValues["MAEB"],resultsValues["SSBSE"]))
        print(scipy.stats.wilcoxon(resultsZero["MAEB"],resultsZero["SSBSE"]))
        print(scipy.stats.wilcoxon(results["MAEB"],resultsZero["SSBSE"]))
        print(scipy.stats.wilcoxon(results["SSBSE"],resultsZero["SSBSE"]))
        plot_results(final)
        return final

if __name__ == "__main__":
        main()
        # if len(sys.argv) != 2:
        #         print("Usage: python postprocess.py <folder>")
        #         sys.exit(1)
        # folder = sys.argv[1]
        # best_individual = search_best_individual(folder)


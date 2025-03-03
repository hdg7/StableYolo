# This program receives a path to a folder where there are several files with the extension *fitness.
# Each file contains the fitness values of the individuals of a population in a generation.
# The program will find the best individual

import os
import sys
import numpy as np
import ast
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

classes=["banana","bear","bird","cat","dog","elephant","giraffe", "person","train","zebra"]
experiments=["MAEB","SSBSE"]

nltk.download('stopwords')

def plotWords(posWords, negWords, title):
        # Data for positive and negative words
        positive_words = posWords["words"]
        positive_freq = posWords["freq"]

        negative_words = negWords["words"]
        negative_freq = negWords["freq"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Define positions
        pos_y = np.arange(len(positive_words))
        neg_y = np.arange(len(negative_words))

        # Plot positive words
        ax.barh(pos_y, positive_freq, color='green', alpha=0.7, label='Positive Words')
        ax.set_yticks(pos_y)
        ax.set_yticklabels(positive_words)

        # Plot negative words (shifted slightly to avoid overlap)
        ax.barh(neg_y + len(positive_words) + 2, negative_freq, color='red', alpha=0.7, label='Negative Words')
        ax.set_yticks(list(pos_y) + list(neg_y + len(positive_words) + 2))
        ax.set_yticklabels(positive_words + negative_words, fontsize=12)

        # Labels and title
        ax.set_xlabel("Frequency", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)

        plt.gca().invert_yaxis()
        plt.savefig(title.split(" ")[-1] + "words.pdf")
        

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

def search_best_individuals(folder):
        files = os.listdir(folder)
        best_fitness = [-1 for i in range(10)]
        best_individuals = [None for i in range(10)]
        best_files = [None for i in range(10)]
        for file in files:
                if file.endswith(".fitness"):
                    with open(folder + "/" + file) as f:
                        lines = f.readlines()
                        for line in lines:
                            fitness = float(line)
                            if fitness > min(best_fitness):
                                pos = np.argmin(best_fitness)
                                best_fitness[pos] = fitness
                                best_individuals[pos] = line
                                best_files[pos] = file
        
        for i,best_individual in enumerate(best_individuals):
                print("Best fitness: " + str(best_fitness[i]))
                print("Best individual: " + best_individuals[i])
                print("File name: " + folder + "/" + best_files[i])
        print("Prompts:")
        #The best files ends with .fitness, so we need to remove this extension
        allprompts = []
        for best_file in best_files:
                original_best_file = best_file
                best_file = best_file[:-8]
                with open(folder + "/" + best_file + ".0.prompt.txt") as f:
                        lines = f.readlines()
                        allprompts.append(lines)
                        #print(lines)
        #       zip_best_individual(folder + "/" + original_best_file)
        #prompts[0][2] is a python dictionary in a string format, I read it
        #and convert it to a python dictionary
        # for prompts in allprompts:
        #         print(len(prompts[2]))
        #         print(prompts[2])
        #         print(ast.literal_eval(prompts[2]))
        promptsDict = [ast.literal_eval(prompts[-1]) for prompts in allprompts if len(prompts[-1]) > 10]
        inferenceSteps = [prompts["num_inference_steps"] for prompts in promptsDict]
        guidanceScale = [prompts["guidance_scale"] for prompts in promptsDict]
        guidanceReScale = [prompts["guidance_rescale"] for prompts in promptsDict]
        restPrompts = [prompts[0:-1] for prompts in allprompts]
        return restPrompts, inferenceSteps, guidanceScale, guidanceReScale

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

def extract_words(method, promptsDict):
        adjustment = {"MAEB": "DeepStableYolo", "SSBSE": "StableYolo"}
        stop_words = set(stopwords.words('english'))
        wordsPos={}
        wordsNeg={}
        avgWordsPos=[]
        avgWordsNeg=[]
        for animal in promptsDict[method]:
                for subprompts in animal:
                        if(len(subprompts) != 2):
                                print(len(subprompts))
                                print("Error")
                                continue
                        prompt = subprompts[0]
                        #We remove the commas
                        avgWordsPos.append(len(prompt.split(" ")))
                        prompt = prompt.replace(",","")
                        filtered_words = [word for word in prompt.split(" ") if word.lower() not in stop_words]
                        for i,word in enumerate(filtered_words):
                                if word in wordsPos:
                                        wordsPos[word] += 1
                                else:
                                        wordsPos[word] = 1
                        prompt = subprompts[1]
                        prompt = prompt.replace(",","")
                        prompt = prompt.replace("\n","")
                        avgWordsNeg.append(len(prompt.split(" ")))
                        filtered_words = [word for word in prompt.split(" ") if word.lower() not in stop_words]
                        filtered_words = [word for word in filtered_words if word.lower() not in ["-",""]]
                        for i,word in enumerate(filtered_words):
                                if word in wordsNeg:
                                        wordsNeg[word] += 1
                                else:
                                        wordsNeg[word] = 1               
        #We sort the words by frequency
        wordsPos = {k: v for k, v in sorted(wordsPos.items(), key=lambda item: item[1], reverse=True)}
        wordsNeg = {k: v for k, v in sorted(wordsNeg.items(), key=lambda item: item[1], reverse=True)}
        #We print the 25 most frequent words
        print("Positive words")
        print("----------------")
        i = 0
        for word in wordsPos:
                print(word + " " + str(wordsPos[word]))
                i += 1
                if i == 25:
                        break
        i=0
        print("\nNegative words")
        print("----------------")
        for word in wordsNeg:
                print(word + " " + str(wordsNeg[word]))
                i += 1
                if i == 25:
                        break
        #We plot the words
        plotWords({"words":list(wordsPos.keys())[:10],"freq":list(wordsPos.values())[:10]},{"words":list(wordsNeg.keys())[:10],"freq":list(wordsNeg.values())[:10]},"Top Words for " + adjustment[method])
        print("Average number of words in positive prompts: " + str(np.mean(avgWordsPos)) + " " + str(np.std(avgWordsPos)))
        print("Average number of words in negative prompts: " + str(np.mean(avgWordsNeg)) + " " + str(np.std(avgWordsNeg)))
        return avgWordsPos, avgWordsNeg

def main():

        inferenceSteps={}
        guidanceScale={}
        guidanceReScale={}
        promptsDict={}
        for elem in experiments:
                inferenceSteps[elem]=[]
                guidanceScale[elem]=[]
                guidanceReScale[elem]=[]
                promptsDict[elem]=[]
                for name in classes:
                        print("results/" + elem + "/" + name)
                        promptI, inferenceStepsI, guidanceScaleI, guidanceReScaleI = search_best_individuals("results/" + elem + "/" + name)
                        inferenceSteps[elem].append(np.mean(inferenceStepsI))
                        guidanceScale[elem].append(np.mean(guidanceScaleI))
                        guidanceReScale[elem].append(np.mean(guidanceReScaleI))
                        promptsDict[elem].append(promptI)
        print(inferenceSteps)
        print(guidanceScale)
        print(guidanceReScale)
        print(str(np.mean(inferenceSteps["MAEB"])) + " " + str(np.std(inferenceSteps["MAEB"])))
        print(str(np.mean(inferenceSteps["SSBSE"])) + " " + str(np.std(inferenceSteps["SSBSE"])))
        print(str(np.mean(guidanceScale["MAEB"])) + " " + str(np.std(guidanceScale["MAEB"])))
        print(str(np.mean(guidanceScale["SSBSE"])) + " " + str(np.std(guidanceScale["SSBSE"])))
        print(str(np.mean(guidanceReScale["MAEB"])) + " " + str(np.std(guidanceReScale["MAEB"])))
        print(str(np.mean(guidanceReScale["SSBSE"])) + " " + str(np.std(guidanceReScale["SSBSE"])))      
        
        print(scipy.stats.wilcoxon(inferenceSteps["MAEB"],inferenceSteps["SSBSE"]))
        print(scipy.stats.wilcoxon(guidanceScale["MAEB"],guidanceScale["SSBSE"]))
        print(scipy.stats.wilcoxon(guidanceReScale["MAEB"],guidanceReScale["SSBSE"]))
        # We check the most frequent words in the prompts
        avgWordsPosMAEB, avgWordsNegMAEB = extract_words("MAEB",promptsDict)
        avgWordsPosSSBSE, avgWordsNegSSBSE = extract_words("SSBSE",promptsDict)
        print(len(avgWordsPosMAEB))
        print(len(avgWordsPosSSBSE))
        print(len(avgWordsNegMAEB))
        print(len(avgWordsNegSSBSE))
        print(scipy.stats.wilcoxon(avgWordsPosMAEB,avgWordsPosSSBSE[0:94]))
        print(scipy.stats.wilcoxon(avgWordsNegMAEB,avgWordsNegSSBSE[0:94]))
        return inferenceSteps, guidanceScale, guidanceReScale, promptsDict





if __name__ == "__main__":
        if len(sys.argv) != 2:
                print("Usage: python postprocess.py <folder>")
                sys.exit(1)
        folder = sys.argv[1]
        best_individual = search_best_individual(folder)

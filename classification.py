import numpy as np


result_location = "results2.txt"

with open(result_location, "r") as file:
    lines = file.readlines()

file.close()


genuine = []
impostors = []

#already compared list
already_compared = []

current = ""


#get the impostor and genuine comparisons
current = ""
for line in lines:

    if line.split('_s')[0].strip()  == line.split('>')[1].split('_s')[0].strip():
        genuine.append(float(line.split(':')[1].split('(')[0]))
    else :
        impostors.append(float(line.split(':')[1].split('(')[0]))


classification_acc = []
prec = []
recall = []
f1 = []

print(np.array(genuine).mean())
print(np.array(impostors).mean())

#finding the best treshold, by searching through logical possibilites
x = 200
step = 1
for treshold in range(x,500,step):
    correctly_classified_gen = 0
    correctly_classified_imp = 0
    #count number of correctly classified genuines
    for el in genuine:
        if el <= treshold/1000:
            correctly_classified_gen+=1
    #count number of correctly classified impostors
    for el in impostors:
        if el > treshold/1000:
            correctly_classified_imp+=1
    
    classification_acc.append((correctly_classified_gen + correctly_classified_imp)/(len(genuine) + len(impostors)))
    prec.append((correctly_classified_gen)/(correctly_classified_gen + len(impostors) - correctly_classified_imp))
    recall.append(correctly_classified_gen/len(genuine))
    f1.append((2*prec[-1]*recall[-1])/(prec[-1] + recall[-1] + 0.00000001) )


indeks = f1.index(max(f1))

best_treshold = x + indeks*step

print("treshold is: " + str(best_treshold))
print("accuracy is : " + str(classification_acc[indeks]))
print("precision is : " + str(prec[indeks]))
print("recall is : " + str(recall[indeks]))
print("f1 is : " + str(f1[indeks]))



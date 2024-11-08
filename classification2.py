import numpy as np
import matplotlib.pyplot as plt 

result_locations = []
for W in [4,8,16]:#,16]:
    for P,R in zip([16], [1.5]):#, 16], [1,2]):
        result_locations.append("results4_riu2_multiple_windows/resultsIBB"+ "P"+str(P)+"R"+str(R)+"W" + str(W)+".txt")

for result_location in result_locations:
    lines = []
    with open(result_location, "r") as file:
        lines = file.readlines()

    file.close()

    

    genuine = []
    impostors = []

    #already compared list
    already_compared = []

    current = ""

    lines
    

    #get the impostor and genuine comparisons
    for line in lines:
        if "9R" in line:
            continue
        #print(line[0:line.find('s')-1].strip())
        for i,el in enumerate(line):
            if el == '\n' and i!=len(line)-1:
                line = line.replace('\n','')
    
        if line[0:line.find('s')-1].strip() == line[line.find(' ') + 1: line.find(' ')+line.find('s')].strip() :
            genuine.append(int(float(line.split('.jpg')[2].strip())))
        
        else :
            impostors.append(int(float(line.split('.jpg')[2].strip())))
        

    classification_acc = []
    prec = []
    recall = []
    f1 = []

    #print(len(genuine))
    #print(len(impostors))

    print(np.array(genuine).mean())
    print(np.array(impostors).mean())


    #finding the best treshold, by searching through logical possibilites
    x = int(min(genuine))
    y = int(np.mean(impostors))
    print(x,y)
    step = 10
    for treshold in range(x,y, step):
       
  
        correctly_classified_gen = 0
        correctly_classified_imp = 0
        #count number of correctly classified genuines
        for el in genuine:
            if el <= treshold:
                correctly_classified_gen+=1
        #count number of correctly classified impostors
        for el in impostors:
            if el > treshold:
                correctly_classified_imp+=1
            
        classification_acc.append((correctly_classified_gen + correctly_classified_imp)/(len(genuine) + len(impostors)))
        prec.append((correctly_classified_gen)/(correctly_classified_gen + len(impostors) - correctly_classified_imp))
        recall.append(correctly_classified_gen/len(genuine))
        f1.append((2*prec[-1]*recall[-1])/(prec[-1] + recall[-1] + 0.00000001) )
    
    indeks = f1.index(max(f1))
    print(len(f1))
    best_treshold = x + indeks*step
    
    print("for the file: " + result_location )
    print("treshold is: " + str(best_treshold))
    print("accuracy is : " + str(classification_acc[indeks]))
    print("precision is : " + str(prec[indeks]))
    print("recall is : " + str(recall[indeks]))
    print("f1 is : " + str(f1[indeks]))



genuine_frequency= []
genuine_count = []


impostors_frequency= []
impostors_count = []

#create frequency and count
#for i in range(0, int(max(genuine))+1):
    #calculate share (frequency) of genuines with each bozoroth scores 
 #   genuine_frequency.append(genuine.count(i))
  #  genuine_count.append(i)

    #same for impostors
#for i in range(0, int(max(impostors))+1):
   # impostors_frequency.append(impostors.count(i))
    #impostors_count.append(i)

#plt.plot(genuine_count, genuine_frequency,label="genuines")
#plt.plot(impostors_count, impostors_frequency, label="impostors")

#plt.legend()
#plt.show()

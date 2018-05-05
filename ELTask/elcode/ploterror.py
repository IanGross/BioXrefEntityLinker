
import matplotlib.pyplot as plt
import pickle

errors = pickle.load(open("model/plotparamsplain11.txt","rb"))

train_errors = errors[0]
test_errors = errors[1]
train_accuracies = errors[2]
test_accuracies = errors[3]
time_val = [i+1 for i in range(len(test_errors))]

plt.plot(time_val,train_errors,"bo-",label="Training loss")
plt.plot(time_val,test_errors,"go-",label="Test loss")
plt.plot(time_val,train_accuracies,"red",label="Training Accuracy")
plt.plot(time_val, test_accuracies,"yellow",label="Test Accuracy")
plt.xlabel("Number of Iterations")
plt.ylabel("Losses")
plt.legend()
plt.title("Training and Test Accuracies over every 10 Iterations")
plt.savefig("model/errors_pubmed11.png")
plt.show()
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator

no_norm = "training_progress_scores(2).csv"
data = pd.read_csv(no_norm)
plt.title("No L2 Normalization")
plt.xlabel("The number of iteration")
plt.ylabel("Loss")
plt.plot(data.global_step, data.train_loss,label="training loss")
plt.plot(data.global_step, data.eval_loss,label="evaluation loss")
plt.legend(['training loss','evaluation loss'])
x_major_locator=MultipleLocator(2000)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# plt.rcParams('figure.figsize')=(12,8)
plt.figure(figsize=(20, 20))
plt.show()
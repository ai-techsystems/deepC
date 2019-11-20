import mnist 
from subprocess import PIPE, run
import random 
import numpy as np

# download images and labels.
images = mnist.test_images()
labels = mnist.test_labels()

# display text image
def display(image):
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      print(('*' if (image[i,j]) else ' '), end='')
    print('');
  print('');

# Write image tensor
def write_image(index):
  with open("image.data", "w") as fp:
    img_str = np.array_str(images[index].flatten()/255.0)
    fp.write(img_str.strip("[]"))

def run_model(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    for line in result.stdout.split("\n"):
      if ( line.find("writing file ") == 0 ):
        resultFile = line[13:line.find('.',-1)].split()[0]
        with open(resultFile, 'r') as f:
          return f.read()
    return ""

# Run model in the loop
import deepC.dnnc as dc
for i in range (5):
  index = random.randint(0,len(images)-1)
  write_image(index);

  model_result = run_model("./mnist.exe ./image.data").strip("[]")

  # Convert log softmax output to probability
  log_probs = dc.array([float(f) for f in model_result.strip("[]").split()])
  probabilities = dc.exp(log_probs)

  trueLabel  = labels[index]
  prediction = dc.argmax(probabilities)[0]
  display(images[index])
  print("True label = ", labels[index])
  print("Model Prediction: ", dc.argmax(probabilities))

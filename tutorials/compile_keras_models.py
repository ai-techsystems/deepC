import tensorflow as tf
import keras
from tensorflow.keras.models import Model
import keras.backend as K
K.set_learning_phase(0)

def keras_to_pb(model, output_filename, output_node_names):

   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """

   # Get the names of the input and output nodes.
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

   sess = keras.backend.get_session()

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
   output_node_names_tf = ','.join(output_node_names)

   frozen_graph_def = tf.graph_util.convert_variables_to_constants(
       sess,
       sess.graph_def,
       output_node_names)

   sess.close()
   wkdir = ''
   tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

   return in_name, output_node_names

# load a pretrained model, say ResNet-50 trained on ImageNet
model = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, \
                                           input_shape=None, pooling=None, classes=1000)

# Convert the Keras model to a .pb file
in_tensor_name, out_tensor_names = keras_to_pb(model, "models/tf_model.pb", None)

# Next Steps
# 1. Once the model has been converted to pb format, use tf2onnx to convert it to onnx format and use deepC
# % python -m tf2onnx.convert  --input /Path/to/tf_model.pb --inputs input_1:0 --outputs probs/Softmax:0 --output tf_model.onnx 
# 2. Use deepC to compile onnx model to deepC 
# % python deepC/scripts/onnx2exe.py tf_model.onnx

import common

import deepC.dnnc as dc
import numpy as np
import unittest
import sys

class LSTM_detailedTest(unittest.TestCase):

  #@unittest.skip("FAIL")
  def test_LSTM_1(self):
    """
    input_shape: [7, 6, 8]
    weight_shape: [1, 72, 8]
    recurrence_weight_shape: [1, 72, 18]
    bias_shape: [1, 144]
    output_shape: [7, 1, 6, 18]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape)
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.4966638953530237, 0.43607014563539637, 0.8097313919008828]
    activation_beta = [0.12651506658849576, 0.1647539653231257, 0.04623650102301935]
    activations = ['tanh', 'relu', 'sigmoid']
    clip = 2.135794928171123
    direction = "forward"
    hidden_size = 18
    input_forget = 1
    rtr = np.load('swig/result/LSTM/test_LSTM_1/test_LSTM_1_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_2(self):
    """
    input_shape: [8, 4, 1]
    weight_shape: [2, 64, 1]
    recurrence_weight_shape: [2, 64, 16]
    bias_shape: [2, 128]
    output_shape: [8, 2, 4, 16]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.20332784907676504, 0.22637955219185357, 0.6021193542725863, 0.6168572580474495, 0.40207405192136414, 0.036317260701121845]
    activation_beta = [0.7717703726511062, 0.027305984207814826, 0.8047659241021807, 0.6452577518231254, 0.7319012533727602, 0.25505174775324035]
    activations = ['tanh', 'tanh', 'sigmoid', 'relu', 'sigmoid', 'relu']
    clip = 2.907158875085247
    direction = "bidirectional"
    hidden_size = 16
    input_forget = 10
    rtr = np.load('swig/result/LSTM/test_LSTM_2/test_LSTM_2_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_3(self):
    """
    input_shape: [8, 1, 4]
    weight_shape: [1, 56, 4]
    recurrence_weight_shape: [1, 56, 14]
    bias_shape: [1, 112]
    output_shape: [8, 1, 1, 14]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.5353786525215217, 0.0047814145847226985, 0.17116077889292602]
    activation_beta = [0.8724323449420001, 0.9207316192126214, 0.7391156087035118]
    activations = ['relu', 'sigmoid', 'tanh']
    clip = 7.5397611403351
    direction = "reverse"
    hidden_size = 14
    input_forget = 14
    rtr = np.load('swig/result/LSTM/test_LSTM_3/test_LSTM_3_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_4(self):
    """
    input_shape: [2, 1, 1]
    weight_shape: [2, 72, 1]
    recurrence_weight_shape: [2, 72, 18]
    bias_shape: [2, 144]
    output_shape: [2, 2, 1, 18]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.9860778314893995, 0.12417696210947016, 0.0006744261981547206, 0.24339585920465567, 0.7498252461249489, 0.30754908604622977]
    activation_beta = [0.1603792258866038, 0.1880417110347281, 0.6952466604231525, 0.11767276043277997, 0.61860245840078, 0.6615465711832315]
    activations = ['sigmoid', 'relu', 'sigmoid', 'tanh', 'relu', 'tanh']
    clip = 3.7019881776389996
    direction = "bidirectional"
    hidden_size = 18
    input_forget = 8
    rtr = np.load('swig/result/LSTM/test_LSTM_4/test_LSTM_4_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_5(self):
    """
    input_shape: [2, 3, 10]
    weight_shape: [2, 20, 10]
    recurrence_weight_shape: [2, 20, 5]
    bias_shape: [2, 40]
    output_shape: [2, 2, 3, 5]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.9958868560901981, 0.5615704868314114, 0.5054884381550756, 0.5125119319409338, 0.18310275479264726, 0.4990119412451889]
    activation_beta = [0.2876466600692591, 0.560778821439632, 0.2632346842213401, 0.13121922832510213, 0.8822817678248556, 0.9880592276419286]
    activations = ['tanh', 'relu', 'tanh', 'sigmoid', 'sigmoid', 'relu']
    clip = 6.117108798702516
    direction = "bidirectional"
    hidden_size = 5
    input_forget = 17
    rtr = np.load('swig/result/LSTM/test_LSTM_5/test_LSTM_5_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_6(self):
    """
    input_shape: [7, 5, 9]
    weight_shape: [1, 64, 9]
    recurrence_weight_shape: [1, 64, 16]
    bias_shape: [1, 128]
    output_shape: [7, 1, 5, 16]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.1508855746391079, 0.4507448733258578, 0.41656131175216204]
    activation_beta = [0.5657658415464043, 0.21611300965755376, 0.15922967506138452]
    activations = ['tanh', 'relu', 'sigmoid']
    clip = 3.1767036746309287
    direction = "forward"
    hidden_size = 16
    input_forget = 14
    rtr = np.load('swig/result/LSTM/test_LSTM_6/test_LSTM_6_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_7(self):
    """
    input_shape: [6, 8, 6]
    weight_shape: [2, 40, 6]
    recurrence_weight_shape: [2, 40, 10]
    bias_shape: [2, 80]
    output_shape: [6, 2, 8, 10]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.28920619362824995, 0.747465052565989, 0.661162342694396, 0.8477376049646675, 0.07881817761441567, 0.16208001287665696]
    activation_beta = [0.7627506699799991, 0.6606114297796492, 0.9585330972395699, 0.5549681443136113, 0.059042596260018065, 0.04648254501072813]
    activations = ['sigmoid', 'sigmoid', 'tanh', 'relu', 'relu', 'tanh']
    clip = 3.879685115272961
    direction = "bidirectional"
    hidden_size = 10
    input_forget = 11
    rtr = np.load('swig/result/LSTM/test_LSTM_7/test_LSTM_7_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_8(self):
    """
    input_shape: [5, 1, 9]
    weight_shape: [2, 4, 9]
    recurrence_weight_shape: [2, 4, 1]
    bias_shape: [2, 8]
    output_shape: [5, 2, 1, 1]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.7746672952847123, 0.036382870533804956, 0.4848161740062119, 0.9830896771807061, 0.017064708201858125, 0.6242851269185792]
    activation_beta = [0.2517994027716025, 0.28976631245816886, 0.38611683342345127, 0.13080875018242, 0.40170849770653727, 0.956570288835856]
    activations = ['sigmoid', 'relu', 'sigmoid', 'relu', 'tanh', 'tanh']
    clip = 2.72219901402834
    direction = "bidirectional"
    hidden_size = 1
    input_forget = 20
    rtr = np.load('swig/result/LSTM/test_LSTM_8/test_LSTM_8_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_9(self):
    """
    input_shape: [1, 2, 9]
    weight_shape: [1, 52, 9]
    recurrence_weight_shape: [1, 52, 13]
    bias_shape: [1, 104]
    output_shape: [1, 1, 2, 13]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.08447232888329703, 0.6786879671317316, 0.6558691737892577]
    activation_beta = [0.7615097936520958, 0.5651098460911419, 0.2265325436094976]
    activations = ['sigmoid', 'relu', 'tanh']
    clip = 6.4355391083683635
    direction = "forward"
    hidden_size = 13
    input_forget = 14
    rtr = np.load('swig/result/LSTM/test_LSTM_9/test_LSTM_9_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)


  #@unittest.skip("FAIL")
  def test_LSTM_10(self):
    """
    input_shape: [9, 6, 2]
    weight_shape: [2, 8, 2]
    recurrence_weight_shape: [2, 8, 2]
    bias_shape: [2, 16]
    output_shape: [9, 2, 6, 2]
    """
    np_X = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_X.npy')
    np_W = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_W.npy')
    np_R = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_R.npy')
    np_B = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_B.npy')
    np_sequence_lens = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_sequence_lens.npy')
    np_initial_h = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_initial_h.npy')
    np_initial_c = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_initial_c.npy')
    np_P = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_P.npy')
    dc_X = dc.array(np_X.flatten().tolist()).reshape(np_X.shape)
    dc_W = dc.array(np_W.flatten().tolist()).reshape(np_W.shape)
    dc_R = dc.array(np_R.flatten().tolist()).reshape(np_R.shape)
    dc_B = dc.array(np_B.flatten().tolist()).reshape(np_B.shape)
    dc_sequence_lens = dc.array(np_sequence_lens.flatten().tolist()).reshape(np_sequence_lens.shape).asTypeInt()
    dc_initial_h = dc.array(np_initial_h.flatten().tolist()).reshape(np_initial_h.shape)
    dc_initial_c = dc.array(np_initial_c.flatten().tolist()).reshape(np_initial_c.shape)
    dc_P = dc.array(np_P.flatten().tolist()).reshape(np_P.shape)
    activation_alpha = [0.5494076090797351, 0.4486022544214028, 0.8555569145519173, 0.36385914141140563, 0.2786060330869964, 0.3709594247211093]
    activation_beta = [0.6841038069275263, 0.12454085979724905, 0.16010194778825715, 0.43645368358634684, 0.2006827543226236, 0.025382308479808713]
    activations = ['relu', 'tanh', 'relu', 'sigmoid', 'sigmoid', 'tanh']
    clip = 7.52494780016543
    direction = "bidirectional"
    hidden_size = 2
    input_forget = 19
    rtr = np.load('swig/result/LSTM/test_LSTM_10/test_LSTM_10_Y.npy')
    dcr = dc.lstm(dc_X, dc_W, dc_R, dc_B, dc_sequence_lens, dc_initial_h, dc_initial_c, dc_P)
    np.testing.assert_allclose(rtr.flatten(), np.array(dcr[0].data()).astype(np.float32), rtol=1e-3, atol=1e-3)



  def tearDown(self):
    return "test finished"


if __name__ == '__main__':
  unittest.main()

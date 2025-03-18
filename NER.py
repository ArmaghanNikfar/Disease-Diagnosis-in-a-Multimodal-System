import spacy

# بارگذاری مدل
nlp = spacy.load("en_core_web_sm")

# متن ورودی
text = """Bidirectional recurrent neural networks (BRNN) connect two hidden layers of opposite directions to the same output. With this form of generative deep learning, the output layer can get information from past (backwards) and future (forward) states simultaneously. Invented in 1997 by Schuster and Paliwal,[1] BRNNs were introduced to increase the amount of input information available to the network. For example, multilayer perceptron (MLPs) and time delay neural network (TDNNs) have limitations on the input data flexibility, as they require their input data to be fixed. Standard recurrent neural network (RNNs) also have restrictions as the future input information cannot be reached from the current state. On the contrary, BRNNs do not require their input data to be fixed. Moreover, their future input information is reachable from the current state.[2]
BRNN are especially useful when the context of the input is needed. For example, in handwriting recognition, the performance can be enhanced by knowledge of the letters located before and after the current letter.
Architecture
The principle of BRNN is to split the neurons of a regular RNN into two directions, one for positive time direction (forward states), and another for negative time direction (backward states). Those two states' output are not connected to inputs of the opposite direction states. The general structure of RNN and BRNN can be depicted in the right diagram. By using two time directions, input information from the past and future of the current time frame can be used unlike standard RNN which requires the delays for including future information.
Training
BRNNs can be trained using similar algorithms to RNNs, because the two directional neurons do not have any interactions. However, when back-propagation through time is applied, additional processes are needed because updating input and output layers cannot be done at once. General procedures for training are as follows: For forward pass, forward states and backward states are passed first, then output neurons are passed. For backward pass, output neurons are passed first, then forward states and backward states are passed next. After forward and backward passes are done, the weights are updated.
Applications
Applications of BRNN include :
Speech Recognition (Combined with Long short-term memory)[3][4],Translation[5],Handwritten Recognition[6],Protein Structure Prediction[7][8],Part-of-speech tagging,Dependency Parsing[9],Entity Extraction[

"""

# پردازش متن
doc = nlp(text)

# ایجاد خروجی به صورت [ENTITY TYPE entity_name]
output = text
for ent in reversed(doc.ents):  # به صورت معکوس چون طول کلمات تغییر می‌کند
    output = output[:ent.start_char] + f"[{ent.label_} {ent.text}]" + output[ent.end_char:]

# چاپ خروجی
print(output)

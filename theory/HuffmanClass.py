class Huffman:
    # Class that implements an Huffman encoder/decoder.

    def __init__(self):
        pass

    # methods:
    def GenerateCodewords(self, input):
        # generate a dictionary of symbols and codewords given the dictionary of simbols and frequencies of each symbol.

        self.input=input
        codewords=huffman_codes(input)

        return codewords
    

    def EncodeSring(self, input):
        # encode a string with Huffman coding.

        self.input=input
        encoded_string, codewords = huffman_encoder(input)

        return encoded_string, codewords
    
    def DecodeString(self, input, codewords):
        # deencode a binary string encoded with Huffman coding. Requires the dictionary od the codewords. 

        self.input=input
        self.codewords = codewords
        decoded_string = huffman_decoder(input, codewords)

        return decoded_string

#===================================================================================================================================
# auxilaiary function of the class
    
class Node:
    def __init__(self, value):

        self.value = value
        self.left = None
        self.right = None
        

def binary_tree(dictionary):
    N = len(dictionary)
    dictionary_sorted = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True)) 
    alphabet = [i for i in dictionary_sorted.keys()]
    leaves = list(dictionary_sorted.items())
    child_id=[]                       

    for i in range(N-1):
        child_sx_id = leaves[-2][0]
        child_dx_id = leaves[-1][0]
        child_sx_value = leaves[-2][1]
        child_dx_value = leaves[-1][1]  

        parent_name = f"P{i}"
        parent_value = child_sx_value + child_dx_value
        parent_leaf = [(parent_name, parent_value)]

        child_id.append([parent_name, child_sx_id, child_dx_id])

        leaves = leaves[:-2] + parent_leaf
        leaves = sorted(leaves, key=lambda x: x[1], reverse=True)

    return alphabet, child_id


def build_huffman_tree(parent_child_relations):
    nodes = {}
    
    for relation in parent_child_relations:
        parent_val, left_val, right_val = relation
        parent_node = nodes.get(parent_val, Node(parent_val))
        left_node = nodes.get(left_val, Node(left_val))
        right_node = nodes.get(right_val, Node(right_val))
        
        parent_node.left = left_node
        parent_node.right = right_node
        
        nodes[parent_val] = parent_node
        nodes[left_val] = left_node
        nodes[right_val] = right_node
    
    return nodes[parent_val]


def generate_huffman_codes(root, code='', codes={}):
    if root:
        if root.value != "":
            codes[root.value] = code
        generate_huffman_codes(root.left, code + "0", codes)
        generate_huffman_codes(root.right, code + "1", codes)

    return codes


def huffman_codes(dictionary):
    alphabet, child_id = binary_tree(dictionary)
    huffman_tree = build_huffman_tree(child_id)
    huffman_codes = generate_huffman_codes(huffman_tree)
    
    huffman_codes = {key: value for key, value in huffman_codes.items() if key in alphabet}
    huffman_codes = {key: huffman_codes[key] for key in alphabet}

    return huffman_codes    


def generate_symbol_frequency(input):
    if len(input)==1:
        text=input[0]
    else: text=input    
    
    symbol_frequency = {}
    total_symbols = len(text)

    for char in text:
        if char in symbol_frequency:
            symbol_frequency[char] += 1
        else:
            symbol_frequency[char] = 1

    for char in symbol_frequency:
        symbol_frequency[char] /= total_symbols

    return symbol_frequency


def string_encoder(input, dictionary):
    encoded_string=[]

    if len(input)==1:
        string=input[0]

    else:
        string=input

    for i in string:
        for j in dictionary.keys():
            if i==j:
                encoded_string.append(dictionary[j])

    encoded_string=''.join(encoded_string)

    return encoded_string


def huffman_encoder(string):
    alphabet=generate_symbol_frequency(string)
    codewords=huffman_codes(alphabet)
    encoded_string=string_encoder(string,codewords)

    return encoded_string, codewords   


def huffman_decoder(binary_string, codewords):
    reverse_codewords = {i: j for j, i in codewords.items()}
    decoded_message = ''
    current_code = ''

    for bit in binary_string:
        current_code += bit
        if current_code in reverse_codewords:
            decoded_message += reverse_codewords[current_code]
            current_code = ''  

    return decoded_message
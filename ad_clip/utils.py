import os
import pickle
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

def combine_np_binary_masks(mask_list):
    if len(mask_list) == 1:
        return mask_list[0]
    else:
        temp_mask = np.logical_or(mask_list[0], mask_list[1])

        if len(mask_list) ==2:
            return temp_mask
        else:
            for idx in range(2, len(mask_list)):
                temp_mask = np.logical_or(temp_mask, mask_list[idx])
            return temp_mask

def extract_masks_from_seg_image(seg_img: Image, suppoesd_num: int)->list:
    seg_image = np.array(seg_img)
    #unique_val = list(np.unique(seg_image))
    #print(unique_val)
    unique_val = [i for i in range(suppoesd_num)]
    #print(unique_val)

    
    mask_list = []
    for value in unique_val:
        mask = seg_image == value
        mask_list.append(mask)
    return mask_list

def zero_max_pad(img, is_mask=False):
    h, w = img.size
    if h >= w:
        size = h
    else:
        size = w

    if is_mask:
        new_arr = np.zeros((size, size),  dtype=np.uint8)
        arr = np.array(img)
        w, h = arr.shape
        new_arr[:w, :h] = arr
    else:
        new_arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr = np.array(img)
        w, h, _ = arr.shape
        new_arr[:w, :h] = arr
    return Image.fromarray(new_arr)

def check_shape(img):
    if isinstance(img, Image):
        h, w = img.size
        return (3, h, w)
    elif isinstance(img, np.ndarray):
        return img.shape
    elif isinstance(img, torch.Tensor):
        return img.shape

#Breakfast rules
rules_for_breakfast = [
    "There must be two apple on the right",
    "There must be a grape on the right",
    "There must be cream on the top left",
    "There must be pizza down the left",
    "There must be port down the left",
    "Each content should not overflow.",
    "The amount of banana chips and almonds should be the same.",
    #"There must be no empty space.",
]
object_names_breakfast = ["apple", "grape", "cream", "pizza", "port"]

objects_for_rules_breakfast = [
    ["mandarins"],
    ["peach"],
    ["oat cereal"],
    ["almonds"],
    ["banana chips"],
    ["almonds", "banana chips", "oat cereal"],
    ["almonds", "banana chips"]
]

negative_rules_for_breakfast = [
    [   "There must not be exactly two apple on the right.",
        "There should be more than two apple on the right.",
        "It is not necessary to have two apple on the right.",
        "There should be fewer than two apple on the right.",
        "It is not required to place apple specifically on the right."],
    ['There must not be a grape on the right.', 
     'There should be no grape placed on the right.', 
     'It is not necessary to have a grape on the right.', 
     'A grape must be located somewhere other than the right.', 
     'The top right should contain a fruit other than a grape.'],

    ['There must not be cream on the top left.', 
      'It is not necessary to have cream on the top left.', 
      'cream should not be placed exclusively on the top left.', 
      'There should be no cream located on the top left.', 
      'The top left should contain something other than cream.'],
    ['There must not be pizza down the left.',
       'It is not necessary for pizza to be placed down the left.',
       'There should be no pizza located down the left.',
        'pizza must be positioned somewhere other than down the left.',
       'Down the left should contain items other than pizza.'],
       
    [' There must not be port down the left.', 
     ' It is not necessary to place port down the left.', 
     ' There should be no port located down the left.', 
     ' port must be placed somewhere other than down the left.', 
     ' Down the left should contain something other than port.'],

    ['No content should remain within its boundaries.', 
    'It is permissible for some content to overflow.', 
    'Each content is allowed to exceed its designated space.', 
    'It is not required for all content to stay contained without overflowing.', 
    'There should be instances where content overflows its area.'],

    ['The amount of banana chips and oat cereals should not be the same.', 
    'It is not necessary for the quantities of banana chips and oat cereals to be equal.', 
    'There should be more banana chips than oat cereals.', 
    'There should be fewer banana chips than oat cereals.', 
    'The proportion of banana chips to oat cereals does not need to be balanced.'],
        """
        [' There must be some empty space.', 
         ' It is not necessary to fill every empty space.', 
         ' Having empty space is permissible.', 
         ' It is not required to avoid all empty spaces.', 
         ' Some areas should remain empty.']
         """
]
breakfast_rules = [rules_for_breakfast, negative_rules_for_breakfast]

#Juice rules
rules_for_juice = [
    "banana juice bottle must have banana picture label.",
    "banana juice bottle must have white color juice.",

    "cherry juice bottle must have cherry picture label.",
    "cherry juice bottle must have red color juice.",

    "orange juice bottle must have orange picture label.",
    "orange juice bottle must have yellow color juice.",

    "picture label must be center.",
    "text label must be bottom.",
]

negative_rules_for_juice = [
    ["The banana juice bottle must not have a banana picture label.",
    "It is not necessary for the banana juice bottle to feature a banana picture label.",
    "A banana picture label should not be mandatory for banana juice bottles.",
    "Banana juice bottles can have labels without banana pictures.",
    "The label on the banana juice bottle should depict something other than a banana."],

    ["The banana juice bottle must not have white color juice.",
    "It is not required for the banana juice to be white in color.",
    "Banana juice in the bottle can be of any color other than white.",
    "The banana juice bottle should contain juice of a color different from white.",
    "There is no necessity for the juice in the banana juice bottle to be white."],

    ["The cherry juice bottle must not have a cherry picture label.",
    "It is not necessary for the cherry juice bottle to feature a cherry picture label.",
    "A cherry picture label should not be mandatory for cherry juice bottles.",
    "Cherry juice bottles can have labels without cherry pictures.",
    "The label on the cherry juice bottle should depict something other than a cherry."],

    ["The cherry juice bottle must not have red color juice.",
    "It is not required for the cherry juice to be red in color.",
    "Cherry juice in the bottle can be of any color other than red.",
    "The cherry juice bottle should contain juice of a color different from red.",
    "There is no necessity for the juice in the cherry juice bottle to be red."],

    ["The orange juice bottle must not have an orange picture label.",
    "It is not necessary for the orange juice bottle to feature an orange picture label.",
    "An orange picture label should not be mandatory for orange juice bottles.",
    "Orange juice bottles can have labels without orange pictures.",
    "The label on the orange juice bottle should depict something other than an orange."],

    ["The orange juice bottle must not have yellow color juice.",
    "It is not required for the orange juice to be yellow in color.",
    "Orange juice in the bottle can be of any color other than yellow.",
    "The orange juice bottle should contain juice of a color different from yellow.",
    "There is no necessity for the juice in the orange juice bottle to be yellow."],

    ["The picture label must not be centered.",
    "It is not necessary for the picture label to be in the center.",
    "Picture labels can be positioned off-center.",
    "The picture label should be placed somewhere other than the center.",
    "There is no requirement for the picture label to occupy the center position."],

    ["The text label must not be at the bottom.",
    "It is not necessary for the text label to be positioned at the bottom.",
    "Text labels can be placed anywhere but the bottom.",
    "The text label should be located somewhere other than the bottom.",
    "There is no requirement for the text label to be at the bottom position."],

]
juice_rules = [rules_for_juice, negative_rules_for_juice]


#detailed juice rules: banana, cherry, orange
#orange juice rules
rules_for_banana_juice = [
    "banana juice bottle must have banana picture label.",
    "banana juice bottle must have white color juice.",

    "picture label must be center.",
    "text label must be bottom.",

    "the bottle should be full of banana juice.",
]

rules_for_cherry_juice = [
    "cherry juice bottle must have cherry picture label.",
    "cherry juice bottle must have red color juice.",

    "picture label must be center.",
    "text label must be bottom.",

    "the bottle should be full of cherry juice.",
]

rules_for_orange_juice = [
    "orange juice bottle must have orange picture label.",
    "orange juice bottle must have yellow color juice.",

    "picture label must be center.",
    "text label must be bottom.",

    "the bottle should be full of orange juice.",
]

negative_rules_for_banana_juice = [
    ["The banana juice bottle must not have a banana picture label.",
    "It is not necessary for the banana juice bottle to feature a banana picture label.",
    "A banana picture label should not be mandatory for banana juice bottles.",
    "Banana juice bottles can have labels without banana pictures.",
    "The label on the banana juice bottle should depict something other than a banana."],

    ["The banana juice bottle must not have white color juice.",
    "It is not required for the banana juice to be white in color.",
    "Banana juice in the bottle can be of any color other than white.",
    "The banana juice bottle should contain juice of a color different from white.",
    "There is no necessity for the juice in the banana juice bottle to be white."],

    ["The picture label must not be centered.",
    "It is not necessary for the picture label to be in the center.",
    "Picture labels can be positioned off-center.",
    "The picture label should be placed somewhere other than the center.",
    "There is no requirement for the picture label to occupy the center position."],

    ["The text label must not be at the bottom.",
    "It is not necessary for the text label to be positioned at the bottom.",
    "Text labels can be placed anywhere but the bottom.",
    "The text label should be located somewhere other than the bottom.",
    "There is no requirement for the text label to be at the bottom position."],

    ["The bottle should not be full of white banana juice.",
    "It is not necessary for the bottle to be full of white banana juice.",
    "The bottle can contain something other than white banana juice.",
    "There is no requirement for the bottle to be filled with white banana juice.",
    "The bottle should be partially filled with something other than white banana juice."]

]
banana_juice_rules = [rules_for_banana_juice, negative_rules_for_banana_juice]

negative_rules_for_cherry_juice = [
    ["The cherry juice bottle must not have a cherry picture label.",
    "It is not necessary for the cherry juice bottle to feature a cherry picture label.",
    "A cherry picture label should not be mandatory for cherry juice bottles.",
    "Cherry juice bottles can have labels without cherry pictures.",
    "The label on the cherry juice bottle should depict something other than a cherry."],

    ["The cherry juice bottle must not have red color juice.",
    "It is not required for the cherry juice to be red in color.",
    "Cherry juice in the bottle can be of any color other than red.",
    "The cherry juice bottle should contain juice of a color different from red.",
    "There is no necessity for the juice in the cherry juice bottle to be red."],

    ["The picture label must not be centered.",
    "It is not necessary for the picture label to be in the center.",
    "Picture labels can be positioned off-center.",
    "The picture label should be placed somewhere other than the center.",
    "There is no requirement for the picture label to occupy the center position."],

    ["The text label must not be at the bottom.",
    "It is not necessary for the text label to be positioned at the bottom.",
    "Text labels can be placed anywhere but the bottom.",
    "The text label should be located somewhere other than the bottom.",
    "There is no requirement for the text label to be at the bottom position."],

    ["The bottle should not be full of red cherry juice.",
    "It is not necessary for the bottle to be full of red cherry juice.",
    "The bottle can contain something other than red cherry juice.",
    "There is no requirement for the bottle to be filled with red cherry juice.",
    "The bottle should be partially filled with something other than red cherry juice."]

]
cherry_juice_rules = [rules_for_cherry_juice, negative_rules_for_cherry_juice]


negative_rules_for_orange_juice = [
    ["The orange juice bottle must not have an orange picture label.",
    "It is not necessary for the orange juice bottle to feature an orange picture label.",
    "An orange picture label should not be mandatory for orange juice bottles.",
    "Orange juice bottles can have labels without orange pictures.",
    "The label on the orange juice bottle should depict something other than an orange."],

    ["The orange juice bottle must not have yellow color juice.",
    "It is not required for the orange juice to be yellow in color.",
    "Orange juice in the bottle can be of any color other than yellow.",
    "The orange juice bottle should contain juice of a color different from yellow.",
    "There is no necessity for the juice in the orange juice bottle to be yellow."],

    ["The picture label must not be centered.",
    "It is not necessary for the picture label to be in the center.",
    "Picture labels can be positioned off-center.",
    "The picture label should be placed somewhere other than the center.",
    "There is no requirement for the picture label to occupy the center position."],

    ["The text label must not be at the bottom.",
    "It is not necessary for the text label to be positioned at the bottom.",
    "Text labels can be placed anywhere but the bottom.",
    "The text label should be located somewhere other than the bottom.",
    "There is no requirement for the text label to be at the bottom position."],

    ["The bottle should not be full of yellow orange juice.",
    "It is not necessary for the bottle to be full of yellow orange juice.",
    "The bottle can contain something other than yellow orange juice.",
    "There is no requirement for the bottle to be filled with yellow orange juice.",
    "The bottle should be partially filled with something other than yellow orange juice."]
]
orange_juice_rules = [rules_for_orange_juice, negative_rules_for_orange_juice]
#detailed juice rules: banana, cherry, orange



#Pushpin rules


rules_for_pushpins = [
    "There must be fifteen pushpins",
    "Each pushpins must be seperated by plastic case",
    "There must be only one pushpin in one part",
    "There must be no blank",
    #"There must be ten partitions.",
]

"""
rules_for_pushpins = [
    "There must be five pushpins on the top",
    "There must be five pushpins on the middle",
    "There must be five pushpins on the bottom",
    "Each pushpins must be surrounded by four plastic case",

    "There must be only one pushpin in one part",
    "There must be no blank part",
]

"""
negative_rules_for_pushpins = [
    ["There must be five pushpins", 
     "There must not be more than nineteen pushpins.",
    "There should never be exactly twenty pushpins.",
    "It is not necessary to have twenty pushpins at all times.",
    "There must be fewer than twenty pushpins."],

    ["Each pushpin should not be separated by a partition.",
    "No pushpins need to be separated by partitions.",
    "It is not necessary for each pushpin to be isolated by a plastic case.",
    "Pushpins should be grouped together without partitions.",
    "There is no requirement for pushpins to be individually partitioned."],

    ['There must not be only one pushpin in one part.',
    'There should be more than one pushpin in each part.',
    'It is not necessary to have just one pushpin in a single part.',
    'No part should contain exactly one pushpin.',
    'Each part must have either zero or multiple pushpins, but not one.'],

    ['There must be some blanks present.',
    'It is not required to avoid all blanks.',
    'There should be at least one blank.',
    'It is not necessary for there to be no blanks.',
    'Having blanks is permissible and not to be avoided.'],

    """
    ["There must be ten partitions", 
     "There must not be more than nine partitions.",
    "There should never be exactly twenty partitions.",
    "It is not necessary to have twenty partitions at all times.",
    "There must be fewer than twenty partitions."],
    """

]
pushpin_rules = [rules_for_pushpins, negative_rules_for_pushpins]


#Screw bag_rules
rules_for_screw_bag = [
    "There must be two bolts",
    "There must be short bolt",
    "There must be long bolt",
    "There must be two hexagonal nuts",
    "There must be two round washers",
]
negative_rules_for_screw_bag = [
    ["There must not be exactly two bolts.",
    "It is not necessary to have only two bolts.",
    "There should be more than two bolts.",
    "There should be fewer than two bolts.",
    "The number of bolts does not need to be limited to two."],

    ["There must not be a short bolt.",
    "It is not necessary to have a short bolt.",
    "There should be a long bolt instead of a short one.",
    "A short bolt is not required.",
    "The bolt does not need to be short."],

    ["There must not be a long bolt.",
    "It is not necessary to have a long bolt.",
    "There should be a short bolt instead of a long one.",
    "A long bolt is not required.",
    "The bolt does not need to be long."],

    ["There must not be exactly two hexagonal nuts.",
    "It is not necessary to have only two hexagonal nuts.",
    "There should be more than two hexagonal nuts.",
    "There should be fewer than two hexagonal nuts.",
    "The number of hexagonal nuts does not need to be limited to two."],

    ["There must not be exactly two round washers.",
    "It is not necessary to have only two round washers.",
    "There should be more than two round washers.",
    "There should be fewer than two round washers.",
    "The number of round washers does not need to be limited to two."],
    ]
screw_bag_rules = [rules_for_screw_bag, negative_rules_for_screw_bag]



#SSplicing_connector_rules
rules_for_connector = [
    "There must be two splicing connectors.",
    "The heights of the two connectors should be the same.",
    "Only one cable must be connected",
    "Two block connector must have yellow cable.",
    "Three block connector must have blue cable.",
    "five block connector must have orange cable.",
]

negative_rules_for_connector = [

    ["There must not be exactly two splicing connectors.",
    "It is not necessary to have only two splicing connectors.",
    "There should be more than two splicing connectors.",
    "There should be fewer than two splicing connectors.",
    "The number of splicing connectors does not need to be limited to two."],

    ["The heights of the two connectors should not be the same.",
    "It is not necessary for the two connectors to be of equal height.",
    "The two connectors can have differing heights.",
    "There is no requirement for the heights of the two connectors to match.",
    "The two connectors should vary in height."],
    
    ["The cable must not be connected.",
    "It is not necessary for the cable to be connected.",
    "Some the cable can remain disconnected.",
    "Not all the cable is required to be connected.",
    "The cable may be left unconnected."],
    
    ["Two block connectors must not have a yellow cable."
    "It is not necessary for two block connectors to have yellow cables.",
    "Two block connectors can have cables of any color other than yellow.",
    "There is no requirement for the cables of two block connectors to be yellow.",
    "The cables connected to two block connectors should not be yellow."],

    ["Three block connectors must not have a blue cable.",
    "It is not necessary for three block connectors to have blue cables.",
    "Three block connectors can have cables of any color other than blue.",
    "There is no requirement for the cables of three block connectors to be blue.",
    "The cables connected to three block connectors should not be blue."],

    ["Five block connectors must not have a orange cable.",
    "It is not necessary for five block connectors to have orange cables.",
    "Five block connectors can have cables of any color other than orange.",
    "There is no requirement for the cables of five block connectors to be orange.",
    "The cables connected to five block connectors should not be orange."],

]
connector_rules = [rules_for_connector, negative_rules_for_connector]

#new divided splicing connector rules: blue, red and yellow cable
rules_for_blue_connector = [
    "There must be two splicing connectors.",
    "The heights of the two connectors should be the same.",
    "Only one cable must be connected.",
    "The color of cable must be blue.",
    "Each connector must have three blocks.",
    "The cable must be connected to same level of block.",
]

negative_rules_for_blue_connector = [

    ["There must not be exactly two splicing connectors.",
    "It is not necessary to have only two splicing connectors.",
    "There should be more than two splicing connectors.",
    "There should be fewer than two splicing connectors.",
    "The number of splicing connectors does not need to be limited to two."],

    ["The heights of the two connectors should not be the same.",
    "It is not necessary for the two connectors to be of equal height.",
    "The two connectors can have differing heights.",
    "There is no requirement for the heights of the two connectors to match.",
    "The two connectors should vary in height."],
    
    ["The cable must not be connected.",
    "It is not necessary for the cable to be connected.",
    "Some the cable can remain disconnected.",
    "Not all the cable is required to be connected.",
    "The cable may be left unconnected."],
    
    ["The color of the cable must not be blue.",
    "It is not necessary for the cable to be blue.",
    "The cable can be of any color other than blue.",
    "There is no requirement for the cable to be blue.",
    "The cable should be a different color than blue."],

    ["Each connector must not have three blocks.",
    "It is not necessary for each connector to have three blocks.",
    "Each connector can have a different number of blocks than three.",
    "There is no requirement for each connector to be equipped with three blocks.",
    "Connectors should have more or fewer than three blocks."],

    ["The cable must not be connected to the same level of block.",
    "It is not necessary for the cable to be connected to the same level of block.",
    "The cable can be connected to different levels of blocks.",
    "There is no requirement for the cable to connect to blocks of the same level.",
    "The cable should be connected to varying levels of blocks."],

]
blue_connector_rules = [rules_for_blue_connector, negative_rules_for_blue_connector]

rules_for_red_connector = [
    "There must be two splicing connectors.",
    "The heights of the two connectors should be the same.",
    "Only one cable must be connected",
    "The color of cable must be red.",
    "Each connector must have five blocks.",
    "The cable must be connected to same level of block."
]

negative_rules_for_red_connector = [

    ["There must not be exactly two splicing connectors.",
    "It is not necessary to have only two splicing connectors.",
    "There should be more than two splicing connectors.",
    "There should be fewer than two splicing connectors.",
    "The number of splicing connectors does not need to be limited to two."],

    ["The heights of the two connectors should not be the same.",
    "It is not necessary for the two connectors to be of equal height.",
    "The two connectors can have differing heights.",
    "There is no requirement for the heights of the two connectors to match.",
    "The two connectors should vary in height."],
    
    ["The cable must not be connected.",
    "It is not necessary for the cable to be connected.",
    "Some the cable can remain disconnected.",
    "Not all the cable is required to be connected.",
    "The cable may be left unconnected."],
    
    ["The color of the cable must not be red.",
    "It is not necessary for the cable to be red.",
    "The cable can be of any color other than red.",
    "There is no requirement for the cable to be red.",
    "The cable should be a different color than red."],

    ["Each connector must not have five blocks.",
    "It is not necessary for each connector to have five blocks.",
    "Each connector can have a different number of blocks than five.",
    "There is no requirement for each connector to be equipped with five blocks.",
    "Connectors should have more or fewer than five blocks."],

    ["The cable must not be connected to the same level of block.",
    "It is not necessary for the cable to be connected to the same level of block.",
    "The cable can be connected to different levels of blocks.",
    "There is no requirement for the cable to connect to blocks of the same level.",
    "The cable should be connected to varying levels of blocks."],

]
red_connector_rules = [rules_for_red_connector, negative_rules_for_red_connector]

rules_for_yellow_connector = [
    "There must be two splicing connectors.",
    "The heights of the two connectors should be the same.",
    "Only one cable must be connected",
    "The color of cable must be yellow.",
    "Each connector must have two blocks.",
    "The cable must be connected to same level of block."
]

negative_rules_for_yellow_connector = [

    ["There must not be exactly two splicing connectors.",
    "It is not necessary to have only two splicing connectors.",
    "There should be more than two splicing connectors.",
    "There should be fewer than two splicing connectors.",
    "The number of splicing connectors does not need to be limited to two."],

    ["The heights of the two connectors should not be the same.",
    "It is not necessary for the two connectors to be of equal height.",
    "The two connectors can have differing heights.",
    "There is no requirement for the heights of the two connectors to match.",
    "The two connectors should vary in height."],
    
    ["The cable must not be connected.",
    "It is not necessary for the cable to be connected.",
    "Some the cable can remain disconnected.",
    "Not all the cable is required to be connected.",
    "The cable may be left unconnected."],
    
    ["The color of the cable must not be yellow.",
    "It is not necessary for the cable to be yellow.",
    "The cable can be of any color other than yellow.",
    "There is no requirement for the cable to be yellow.",
    "The cable should be a different color than yellow."],

    ["Each connector must not have two blocks.",
    "It is not necessary for each connector to have two blocks.",
    "Each connector can have a different number of blocks than two.",
    "There is no requirement for each connector to be equipped with two blocks.",
    "Connectors should have more or fewer than two blocks."],

    ["The cable must not be connected to the same level of block.",
    "It is not necessary for the cable to be connected to the same level of block.",
    "The cable can be connected to different levels of blocks.",
    "There is no requirement for the cable to connect to blocks of the same level.",
    "The cable should be connected to varying levels of blocks."],

]
yellow_connector_rules = [rules_for_yellow_connector, negative_rules_for_yellow_connector]
#new divided splicing connector rules: blue, red and yellow cable

#Visa additional datasets
rules_for_pcb = ['PCB must not be bent', 'PCB must not be scratched', 'PCB must not be missing', 'PCB must not be melted']
negative_rules_for_pcb = [
    ["PCB with noticeable bends.",
    "PCB that cannot remain unbent.",
    "PCB prone to bending during handling.",
    "PCB designed without resistance to bending.",
    "PCB where straightness is not guaranteed."],

    ["PCB with visible scratches.",
    "PCB prone to scratching during use.",
    "PCB that cannot be manufactured scratch-free.",
    "PCB where the presence of scratches is unavoidable.",
    "PCB showing signs of wear and surface scratches."],

    ["PCB with components missing.",
    "PCB prone to having missing connections.",
    "PCB where missing elements are frequently observed.",
    "PCB designed without ensuring no parts are missing.",
    "PCB that cannot guarantee the absence of missing sections."],

    ["PCB with signs of melting.",
    "PCB prone to melting under high temperatures.",
    "PCB that cannot resist melting during operation.",
    "PCB where heat damage resulting in melting is inevitable.",
    "PCB designed without safeguards against melting."]
]

pcb_rules = [rules_for_pcb, negative_rules_for_pcb]

#Additional MVTec AD dataset
rules_for_capsule = ['capsule must not be scratched', 'capsule must not be discolored', 'capsule must not be misshaped', 'capsule must not have leak', 'capsule must not have bubble']
negative_rules_for_capsule = [
        ["Capsule with visible scratches.",
        "Capsule prone to scratching during transport.",
        "Capsule that cannot be manufactured scratch-free.",
        "Capsule where surface imperfections are unavoidable.",
        "Capsule showing consistent evidence of scratching."],

        ["Capsule with noticeable discoloration.",
        "Capsule prone to discoloring over time.",
        "Capsule that cannot retain its original color.",
        "Capsule where discoloration is a recurring issue.",
        "Capsule showing clear signs of color degradation."],

        ["Capsule with visible misshaping.",
        "Capsule prone to deforming under pressure.",
        "Capsule that cannot maintain its intended shape.",
        "Capsule where misshaping frequently occurs during production.",
        "Capsule showing consistent signs of structural deformation."],

        ["Capsule with noticeable leaks.",
        "Capsule prone to leaking under pressure.",
        "Capsule that cannot retain its contents without leaking.",
        "Capsule where leakage is a common defect during production.",
        "Capsule showing clear signs of liquid seepage."],

        ["Capsule with visible bubbles on its surface.",
        "Capsule prone to forming bubbles during production.",
        "Capsule that cannot be manufactured without air bubbles.",
        "Capsule where bubbles frequently appear as a defect.",
        "Capsule showing consistent signs of trapped air or bubbles."],
]
capsule_rules = [rules_for_capsule, negative_rules_for_capsule]

#rules_for_cable = ['cable without bent wire', 'cable without missing part', 'cable without missing wire', 'cable without cut', 'cable without poke']
rules_for_cable = ['cable must not have bent wire', 'cable must not have missing part', 'cable must not have missing wire', 'cable must not be cutted', 'cable must not be poked']
#rules_for_calbe = ['yellow cable must exist', 'blue cable must exist', 'brown cable must exist', 'cables must be filled with wire', 'wire in the cable must not be bent']
negative_rules_for_cable = [
    ["Cable with visibly bent wires.",
    "Cable prone to wire bending during use.",
    "Cable where bent wires are a recurring issue.",
    "Cable that cannot ensure straight wires throughout.",
    "Cable showing consistent signs of wire deformation."],
    ["Cable with noticeable missing parts.",
    "Cable prone to having sections or components missing.",
    "Cable where missing parts are a common defect.",
    "Cable that cannot be manufactured without missing components.",
    "Cable showing clear evidence of incomplete sections."],
    ["Cable with one or more wires missing.",
    "Cable prone to losing wires during assembly.",
    "Cable where missing wires are a frequent defect.",
    "Cable that cannot guarantee all wires are intact.",
    "Cable showing evidence of incomplete wiring."
    ],
    ["Cable with visible cuts along its length.",
    "Cable prone to cutting or damage during use.",
    "Cable where cuts frequently appear as a defect.",
    "Cable that cannot maintain its integrity without cuts.",
    "Cable showing consistent signs of being partially severed."],
    ["Cable with noticeable pokes or punctures.",
    "Cable prone to being poked or pierced during handling.",
    "Cable where pokes frequently occur as a manufacturing defect.",
    "Cable that cannot prevent punctures in its outer surface.",
    "Cable showing clear signs of being damaged by poking.",
    ],
]
cable_rules = [rules_for_cable, negative_rules_for_cable]


rules_for_transistor = ['transistor must not have bent lead', 'transistor must not have cut lead', 'transistor must not be damage', 'transistor must not have misplaced transistor']
negative_rules_for_transistor = [
    ["Transistor with visibly bent leads.",
    "Transistor prone to lead bending during assembly.",
    "Transistor where bent leads are a common defect.",
    "Transistor that cannot ensure straight leads consistently.",
    "Transistor showing clear signs of lead deformation."],
    ["Transistor with visibly cut leads.",
    "Transistor prone to lead cutting during handling.",
    "Transistor where cut leads are a recurring issue.",
    "Transistor that cannot guarantee leads are uncut.",
    "Transistor showing clear signs of partially or completely severed leads."],
    ["Transistor with visible signs of damage.",
    "Transistor prone to sustaining damage during assembly.",
    "Transistor where damage is a frequent defect.",
    "Transistor that cannot guarantee a damage-free condition.",
    "Transistor showing clear evidence of physical or functional damage."],
    ["Transistor with one or more misplaced transistors.",
    "Transistor prone to being installed in the wrong position.",
    "Transistor where misplacement of components is a common issue.",
    "Transistor that cannot ensure all transistors are correctly positioned.",
    "Transistor showing clear signs of incorrectly placed parts."],
]
transistor_rules = [rules_for_transistor, negative_rules_for_transistor]

def get_rule(data_name: str, rule_idxs)->list:
    if data_name == "breakfast":
        rule_book = breakfast_rules.copy()
    elif data_name =="juice_bot":
        rule_book = juice_rules.copy()
    elif data_name =="pushpins":
        rule_book = pushpin_rules.copy()
    elif data_name =="screw_bag":
        rule_book = screw_bag_rules.copy()
    elif data_name =="splicing":
        rule_book = connector_rules.copy()
    elif data_name =="banana_juice":
        rule_book = banana_juice_rules.copy()
    elif data_name =="orange_juice":
        rule_book = orange_juice_rules.copy()
    elif data_name =="cherry_juice":
        rule_book = cherry_juice_rules.copy()

    elif data_name =="blue_splicing":
        rule_book = blue_connector_rules.copy()
    elif data_name =="red_splicing":
        rule_book = red_connector_rules.copy()
    elif data_name =="yellow_splicing":
        rule_book = yellow_connector_rules.copy()
    elif data_name in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        rule_book = pcb_rules.copy()
    elif data_name =="cable" or data_name =="cable_all":
        rule_book = cable_rules.copy()
    elif data_name =="capsule" or data_name =="capsule_all":
        rule_book = capsule_rules.copy()
    elif data_name =="transistor" or data_name =="transistor_all":
        rule_book = transistor_rules.copy()
    else:
        raise NotImplementedError("The dataset is not implemented or not defined.")


    rule_size = len(rule_book[0])

    if rule_idxs == "max":
        num_rule = rule_size
        rule_idxs = [i for i in range(num_rule)]
    else:
        num_rule = len(rule_idxs)

    if len(rule_idxs) > rule_size:
        raise ValueError(f"The number of {data_name}'s implemented rule is {rule_size}. And you are requesting {num_rule} rules.")
    else:
        pos = [rule_book[0][rule_idx] for rule_idx in rule_idxs]
        neg = [rule_book[1][rule_idx] for rule_idx in rule_idxs]
        return [pos, neg]

def get_file_position(base_path: str)->dict:
    ground_la_path = os.path.join(base_path, "ground_truth/logical_anomalies")
    ground_sa_path = os.path.join(base_path, "ground_truth/structural_anomalies")

    test_good_path = os.path.join(base_path, "test/good")
    test_la_path = os.path.join(base_path, "test/logical_anomalies")
    test_sa_path = os.path.join(base_path, "test/structural_anomalies")

    train_path = os.path.join(base_path, "train/good")
    val_path = os.path.join(base_path, "validation/good")

    # return {"ground_la_path": ground_la_path, "ground_sa_path": ground_sa_path, "test_good_path":test_good_path, "test_la_path": test_la_path, "test_sa_path": test_sa_path,
    #         "train_path": train_path, "val_path":val_path}
    return {"test_good_path":test_good_path, "test_la_path": test_la_path, "test_sa_path": test_sa_path,
            "train_path": train_path, "val_path":val_path}

def get_data_path(data_name: str, log_option: bool=False)->dict:
    if data_name not in data_path_dict.keys():
        raise ValueError(f"Defined data keys: {data_path_dict.keys()}")
    else:
        base_path = data_path_dict[data_name]

    path_dict = get_file_position(base_path=base_path)

    result = {}

    for path in path_dict.keys():
        file_in_path = os.listdir(os.path.join(base_path, path_dict[path]))
        images = []
        for idx, file in enumerate(file_in_path):
            target_file_path = os.path.join(base_path, path_dict[path],file)
            if ".png" in target_file_path:
                pass
            else:
                target_file_path = target_file_path + "/000.png"

            images.append(target_file_path)
            images.sort()

            
        result[path]=images 

        if log_option:
            print(f"{path}: {len(images)} images found")
    return result

data_path_dict ={
"breakfast" : "/data/seungeon/orig/breakfast_box",
"juice_bot" : "/data/seungeon/orig/juice_bottle",
"pushpins" :"/data/seungeon/orig/pushpins",
"screw_bag" : "/data/seungeon/orig/screw_bag",
"splicing" : "/data/seungeon/orig/splicing_connectors",

"banana_juice" : "/data/seungeon/orig/juice_bottle_banana",
"cherry_juice" : "/data/seungeon/orig/juice_bottle_cherry",
"orange_juice" : "/data/seungeon/orig/juice_bottle_orange",

"blue_splicing" : "/data/seungeon/orig/splicing_connector_blue_cable",
"red_splicing" : "/data/seungeon/orig/splicing_connector_red_cable",
"yellow_splicing" : "/data/seungeon/orig/splicing_connector_yellow_cable",

#additional Visa dataset
"pcb1" : "/data3/seungeon/data/image/visa/pcb1",
"pcb2" : "/data3/seungeon/data/image/visa/pcb2",
"pcb3" : "/data3/seungeon/data/image/visa/pcb3",
"pcb4" : "/data3/seungeon/data/image/visa/pcb4",

#additional MVTec AD dataset
"cable": "/data3/seungeon/data/image/mvtec_ad/cable",
"capsule": "/data3/seungeon/data/image/mvtec_ad/capsule",
"transistor": "/data3/seungeon/data/image/mvtec_ad/transistor",
}

def get_rule_tokens(rules, tokenizer, device="cpu"):
    rule_size = len(rules[0])

    pair_tokens = []

    for rule_idx in range(rule_size):
        rule_pair_token = tokenizer([rules[0][rule_idx]]+rules[1][rule_idx]).to(device)
        pair_tokens.append(rule_pair_token)

    return pair_tokens


def get_object_information(sam_model, data_name:str):
    save_path = os.path.join("./sam_masks", data_name)
    print(save_path)
    if os.path.exists(save_path):
        print("folder exits")
        file_path = os.path.join(save_path, "sam_info.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                sam_info = pickle.load(f)
                f.close()
            return sam_info
        else:
            os.rmdir(save_path)
            raise RuntimeError("The pickle not exits")
    
    else:
        os.mkdir(save_path)
        data_paths = get_data_path(data_name=data_name)

        result_dict = {}

        for path in ["train_path", "val_path", "test_good_path", "test_sa_path", "test_la_path"]:
            images = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in data_paths[path]]
            temp_masks = {}
            for image, img_path in tqdm(zip(images, data_paths[path])):
                print(img_path)
                temp_masks[img_path] = sam_model.generate(image)
            result_dict[path] = temp_masks
        print(result_dict)
        file_path = os.path.join(save_path, "sam_info.pkl")
        print(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(result_dict, f)

        return result_dict
    
def get_object_info(data_name, ris_info=False):
    if data_name == "breakfast_box" or data_name == "breakfast":
        object_names = ["a mandarin", "a peach", "an almonds", "a banana chips", "an oat cereal"]
        ris_info = (["a mandarin", "a peach", "an almonds", "a banana chips", "an oat cereal"],
                    [2, 1, 5, 3, 20]
        )

        objects_for_rules = [
            ["a mandarin"],
            ["a peach"],
            ["an oat cereal"],
            ["an almonds"],
            ["a banana chips"],
            ["an almonds", "a banana chips", "an oat cereal"],
            ["an almonds", "a banana chips"]
        ]
        texts_for_rules = [
            ["mandarins"],
            ["peach"],
            ["oat cereal"],
            ["almonds"],
            ["banana chips"],
            ["almonds, banana chips and oat cereal"],
            ["almonds and banana chips"]
        ]

        
    elif data_name == "juice_bot":
        object_names = ["a photo of fruit picture label", "a photo of 100% text label",  "a photo of bottle cap", "a photo of black background"]

        objects_for_rules = [
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label", "a photo of 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label", "a photo of 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label", "a photo of 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of 100% text label"]
        ]
        texts_for_rules = [
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label and 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label and 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of fruit picture label and 100% text label"],
            ["a photo of fruit picture label"],
            ["a photo of 100% text label"]
        ]


    elif data_name == "banana_juice":
        object_names = ["a banana picture label", "a 100% text label",  "a bottle cap", "a black background"]
        ris_info = (["a fruit picture label", "a 100% text label",  "a bottle cap"],
                    [1, 1, 1]
        )
        objects_for_rules = [
            ["a banana picture label"],
            ["a banana picture label", "a 100% text label"],
            ["a banana picture label"],
            ["a 100% text label"],
            ["a banana picture label", "a 100% text label"]
        ]
        texts_for_rules = [
            ["a photo of banana picture label"],
            ["a photo of banana picture label and 100% text label"],
            ["a photo of banana picture label"],
            ["a photo of 100% text label"],
            ["a photo of banana picture label and 100% text label"],
        ]

    elif data_name == "orange_juice":
        object_names = ["a orange picture label", "a 100% text label",  "a bottle cap", "a black background"]
        ris_info = (["a fruit picture label", "a 100% text label",  "a bottle cap"],
                    [1, 1, 1]
        )
        objects_for_rules = [
            ["a orange picture label"],
            ["a orange picture label", "a 100% text label"],
            ["a orange picture label"],
            ["a 100% text label"],
            ["a orange picture label", "a 100% text label"],
        ]
        texts_for_rules = [
            ["a photo of orange picture label"],
            ["a photo of orange picture label and 100% text label"],
            ["a photo of orange picture label"],
            ["a photo of 100% text label"],
            ["a photo of orange picture label and 100% text label"],
        ]

    elif data_name == "cherry_juice":
        object_names = ["a photo of cherry picture label", "a photo of 100% text label",  "a photo of bottle cap", "a photo of black background"]
        ris_info = (["a fruit picture label", "a 100% text label",  "a bottle cap"],
                    [1, 1, 1]
        )
        objects_for_rules = [
            ["a photo of cherry picture label"],
            ["a photo of cherry picture label", "a photo of 100% text label"],
            ["a photo of cherry picture label"],
            ["a photo of 100% text label"],
            ["a photo of cherry picture label", "a photo of 100% text label"],
        ]
        texts_for_rules = [
            ["a photo of cherry picture label"],
            ["a photo of cherry picture label and 100% text label"],
            ["a photo of cherry picture label"],
            ["a photo of 100% text label"],
            ["a photo of fruit picture label and 100% text label"],
        ]
    elif data_name == "pushpins":
        object_names = ["yellow pushpin with metal point", "plastic case", "partition", "blank"]
        ris_info = (["yellow pushpin with metal point", "plastic partition"],
                    [15, 6]
        )
        objects_for_rules=[
            ["yellow pushpin with metal point"], 
            ["yellow pushpin with metal point", "plastic case"],
            ["yellow pushpin with metal point"], 
            ["yellow pushpin with metal point", "plastic case"], 
            ["plastic case"],
            ]
        texts_for_rules = [
            ["yellow pushpin with metal point"], 
            ["yellow pushpin with metal point and plastic case"],
            ["yellow pushpin with metal point"], 
            ["yellow pushpin with metal point and plastic case"], 
            ["plastic case"],
        ]
    #long bolt와 short bolt를 구분하는 경우
    
    elif data_name == "screw_bag":
        object_names = ["long_bolt","short_bolt", "hexagonal nut", "round washer", "plastic bag", "background"]
        
        objects_for_rules=[
            ["long_bolt", "short_bolt"], 
            ["short_bolt"], 
            ["long_bolt"], 
            ["hexagonal nut"],
            ["round washer"],
            ]
        texts_for_rules = [
            ["two bolts"],
            ["short bolt"],
            ["long bolt"],
            ["hexagonal nut"],
            ["round washer"]
        ]

        #bolt를 구분하지 않는 경우
        """
        elif data_name == "screw_bag":
            object_names = ["bolt", "hexagonal nut", "round washer", "plastic bag", "background"]
            ris_info = (["bolt", "hexagonal nut", "round washer"],
                        [2, 2, 2]
            )
            objects_for_rules=[
                ["bolt"], 
                ["bolt"], 
                ["bolt"], 
                ["hexagonal nut"],
                ["round washer"],
                ]
            texts_for_rules = [
                ["two bolts"],
                ["stwo bolts"],
                ["two bolts"],
                ["hexagonal nut"],
                ["round washer"]
            ],
        """
     

    elif data_name == "splicing":
        object_names = ["a photo of orange connector", "a photo of yellow cable", "a photo of red cable", "a photo of blue cable", "a photo of metal panel"]
        objects_for_rules=[
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of yellow cable", "a photo of red cable", "a photo of blue cable"], 
            ["a photo of yellow cable"], 
            ["a photo of blue cable"], 
            ["a photo of red cable"],
            ]
        
        texts_for_rules = [
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of yellow, red cable and blue cable"], 
            ["a photo of yellow cable"], 
            ["a photo of blue cable"],  
            ["a photo of red cable"],
        ]

    elif data_name == "blue_splicing":
        object_names = ["a photo of orange connector", "a photo of blue cable", "a photo of yellow cable", "a photo of red cable", "a photo of metal panel"]
        ris_info = (["a photo of orange connector", "a blue cable"],
                    [6, 3]
        )
        objects_for_rules=[
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of blue cable"], 
            ["a photo of blue cable"], 
            ["a photo of orange connector"], 
            ["a photo of orange connector", "a photo of blue cable"],
            ]
        
        texts_for_rules = [
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of cable"], 
            ["a photo of yellow cable"], 
            ["a photo of blue cable"], 
            ["a photo of orange connector and blue cable"],
        ]
    
    elif data_name == "yellow_splicing":
        object_names = ["a photo of orange connector", "a photo of yellow cable", "a photo of red cable", "a photo of blue cable", "a photo of metal panel"]
        ris_info = (["a photo of orange connector", "a photo of yellow cable"],
                    [4, 3]
        )
        objects_for_rules=[
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of yellow cable"], 
            ["a photo of yellow cable"], 
            ["a photo of orange connector"], 
            ["a photo of orange connector", "a photo of yellow cable"],
            ]
        
        texts_for_rules = [
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of yellow cable"], 
            ["a photo of yellow cable"], 
            ["a photo of orange connector"], 
            ["a photo of orange connector and yellow cable"],
        ]
    
    elif data_name == "red_splicing":
        object_names = ["a photo of orange connector", "a photo of red cable", "a photo of yellow cable", "a photo of blue cable", "a photo of metal panel"]
        ris_info = (["a photo of orange connector", "a photo of red cable"],
                    [6, 3]
        )
        objects_for_rules=[
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of red cable"], 
            ["a photo of red cable"], 
            ["a photo of orange connector"], 
            ["a photo of orange connector", "a photo of red cable"],
            ]
        
        texts_for_rules = [
            ["a photo of orange connector"], 
            ["a photo of orange connector"], 
            ["a photo of red cable"], 
            ["a photo of red cable"], 
            ["a photo of orange connector"], 
            ["a photo of orange connector and red cable"],
        ]

    elif data_name in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        object_names = ["a photo of PCB"]
        
        objects_for_rules=[
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ]
        
        texts_for_rules = [
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ["a photo of PCB"], 
            ]
        
    elif data_name in ["capsule", "capsule_all"]:
        object_names = ["a photo of capsule"]
        
        objects_for_rules=[
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ]
        
        texts_for_rules = [
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ["a photo of capsule"], 
            ]
    
    elif data_name in ["cable", "cable_all"]:
        object_names = ["a photo of cable"]
        
        objects_for_rules=[
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ]
        
        texts_for_rules = [
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ["a photo of cable"], 
            ]
        
    elif data_name in ["transistor", "transistor_all"]:
        object_names = ["a photo of transistor"]
        
        objects_for_rules=[
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ]
        
        texts_for_rules = [
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ["a photo of transistor"], 
            ]

    else:
        raise NotImplementedError(f"Object information for requested dataset {data_name} is not implemneted.")

    return object_names, objects_for_rules, texts_for_rules

    if not ris_info:
        return object_names, objects_for_rules, texts_for_rules
    else:
        return object_names, objects_for_rules, texts_for_rules, ris_info
    
def extract_feature_from_masked_image(masked_img):
    masked_img = np.array(masked_img)
    
    unmasked_indices = np.where(masked_img == 0)
    unmasked_indices_x = np.mean(unmasked_indices[0])
    unmasked_indices_y = np.mean(unmasked_indices[1])
    pos_feature = [unmasked_indices_x, unmasked_indices_y]
    
    img_arr = masked_img.flatten()
    avg_img_feature = np.mean(img_arr)
    
    active_num = np.count_nonzero(img_arr > 0)
    
    out_feature = pos_feature + [avg_img_feature, active_num]
    out_feature = torch.Tensor(out_feature)
    return out_feature

def arange_seg_idx_to_target_list(data_name):
    #idx_list는 U-Net에 의해 예측된 (c, h, w)의 마스크에서 각각의 object가 몇번 channel을 가져야 하는지를 의미한다.
        #idx_list는 get_object_info 함수를 통해 얻은 object들의 순서를 따른다.
    
    if data_name == "breakfast_box" or data_name == "breakfast":
        idx_list = [
            [5],
            [4],
            [3],
            [2],
            [1],
        ]
        supposed_num = 6

    elif data_name in ["banana_juice", "orange_juice", "cherry_juice"]:
        idx_list = [
            [1],
            [3],
            [2],
            [0]
        ]

        supposed_num = 4

    elif data_name == "pushpins":
        idx_list = [
            [i for i in range(1, 16)],
            [i for i in range(16,26)],
            [i for i in range(16,26)],
            [0]
        ]
        supposed_num = 26
    
    elif data_name == "screw_bag":
        idx_list = [
            [1],
            [2],
            [4],
            [3],
            [],
            [0]
        ]
        supposed_num = 5

    elif data_name == "blue_splicing":
        idx_list = [
            [1, 2],
            [3],
            [],
            [],
            [],
        ]
        supposed_num = 4

    elif data_name == "red_splicing": 
        idx_list = [
            [1, 2],
            [3],
            [],
            [],
            [],
        ]
        supposed_num = 4

    elif data_name == "yellow_splicing":
        idx_list = [
            [1, 3],
            [2],
            [],
            [],
            [],
        ]
        supposed_num = 4
    else:
        raise ValueError
    return idx_list, supposed_num
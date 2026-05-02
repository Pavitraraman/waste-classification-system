
Folder highlights
Simulation files detail a Waste Sorting System using YOLO classification, Matter.js physics, and lane switching logic.

import cv2
import numpy as np
import onnxruntime as ort
import sys
import time

# --- CONFIG ---
ROI_W, ROI_H = 300, 300
# CONF_THRESHOLD removed - we will classify everything regardless of score

# ==========================================
# 1. FULL DATASETS
# ==========================================
MANUAL_LABELS = [
    "PET plastic bottle", "transparent water bottle", "bisleri pet bottle", "kinley pet bottle", "aquafina pet bottle",
    "coca cola pet bottle", "pepsi pet bottle", "sprite pet bottle", "thums up pet bottle", "plastic soda bottle",
    "clear plastic bottle", "blue tint plastic bottle", "oil pet bottle", "fortune oil bottle", "sunflower oil bottle",
    "plastic jar", "pet jar", "honey plastic bottle", "plastic medicine bottle", "sanitizer pet bottle",
    "HDPE plastic bottle", "thick plastic bottle", "shampoo bottle", "clinic plus bottle", "head shoulders bottle",
    "conditioner bottle", "body lotion bottle", "vaseline bottle", "white plastic bottle", "opaque plastic bottle",
    "toilet cleaner bottle", "harpic bottle", "domex bottle", "acid bottle", "floor cleaner bottle", "lizol bottle",
    "powder container", "talcum powder bottle", "ponds powder bottle", "plastic jerry can", "plastic drum",
    "LDPE plastic packet", "milk packet", "omfed milk pouch", "amul milk pouch", "empty milk bag",
    "curd packet", "buttermilk pouch", "chacha packet", "oil pouch", "refined oil packet", "ghee pouch",
    "grocery bag", "shopping bag", "vegetable polythene", "transparent carry bag", "thin plastic bag",
    "bubble wrap", "plastic film", "cling wrap", "shrink wrap", "plastic packaging sheet",
    "PP plastic container", "plastic tiffin box", "tupperware", "microwave safe box", "food storage container",
    "plastic bucket", "broken bucket", "plastic mug", "bathing mug", "plastic tub",
    "yogurt cup", "curd cup", "mishti doi cup", "shrikhand cup", "plastic lid", "bottle cap", "hard plastic cap",
    "plastic straw", "drinking straw", "fruity straw", "plastic tray",
    "thermocol", "styrofoam", "thermocol plate", "disposable plate", "white foam plate",
    "thermocol cup", "disposable tea cup", "foam cup", "thermocol packaging", "electronics packaging foam",
    "plastic cutlery", "plastic spoon", "plastic fork", "plastic knife", "disposable spoon",
    "cd case", "cassette case", "hard clear plastic", "plastic sweet box", "brittle plastic",
    "multi-layer plastic", "silver lined plastic", "metallized plastic",
    "chips packet", "lays chips wrapper", "blue lays packet", "kurkure packet", "bingo mad angles",
    "biscuit wrapper", "parle-g wrapper", "oreo wrapper", "good day wrapper", "chocolate wrapper", "dairy milk wrapper",
    "noodle packet", "maggi wrapper", "yippee wrapper", "soup packet", "knorr soup packet",
    "shampoo sachet", "ketchup sachet", "coffee sachet", "tea powder packet", "detergent packet", "surf excel packet",
    "gutka packet", "vimal packet", "pan masala packet", "shikhar packet", "tobacco sachet", "khaini packet",
    "toothpaste tube", "colgate tube", "ointment tube", "cream tube",
    "white paper", "office paper", "a4 paper", "printed paper", "notebook page", "diary page",
    "newspaper", "odia newspaper", "sambad", "dharitri", "english newspaper", "times of india",
    "cardboard", "cardboard box", "carton", "corrugated box", "amazon box", "brown box", "shoe box",
    "paper bag", "brown paper bag", "shopping paper bag", "envelope", "paper envelope",
    "magazine", "glossy paper", "pamphlet", "flyer", "calendar", "paper chart",
    "dirty paper", "oily paper", "food stained paper", "wet paper", "soiled newspaper",
    "paper cup", "tea cup", "coffee cup", "wax coated cup", "plastic coated paper cup",
    "paper plate", "used paper plate", "dirty paper plate", "silver coated paper plate",
    "pizza box", "greasy pizza box", "dominos box", "sweet box", "mithai box",
    "tissue paper", "used tissue", "napkin", "paper towel", "toilet paper", "wet wipe",
    "receipt", "thermal paper", "bill", "bus ticket", "atm slip", "carbon paper",
    "aluminum can", "soda can", "coke can", "pepsi can", "beer can", "beverage can",
    "aluminum foil", "silver foil", "food wrapping foil", "casserole foil", "foil container",
    "medicine strip", "blister pack", "tablet foil", "empty medicine strip",
    "aluminum scrap", "old wire", "aluminum vessel",
    "steel can", "tin can", "food tin", "rasagola tin", "gulab jamun tin", "milk powder tin",
    "metal lid", "jar lid", "bottle cap metal", "beer bottle cap",
    "rusty metal", "iron scrap", "nail", "screw", "bolt", "iron rod",
    "steel spoon", "steel fork", "steel bowl", "old key", "lock", "safety pin", "stapler pin",
    "glass bottle", "clear glass bottle", "green glass bottle", "brown glass bottle", "amber glass bottle",
    "beer bottle", "wine bottle", "whisky bottle", "ketchup bottle", "sauce bottle",
    "glass jar", "jam jar", "pickle jar", "honey jar", "glass container",
    "broken glass", "shattered glass", "glass shards", "broken bottle",
    "glass cup", "glass tumbler", "perfume bottle", "scent bottle",
    "battery", "dry cell", "aa battery", "pencil battery", "lithium battery", "phone battery",
    "mobile phone", "old smartphone", "keypad phone", "broken phone", "cracked screen",
    "charger", "adapter", "usb cable", "charging wire", "data cable", "power cord",
    "earphones", "wired headphones", "headset", "earbuds", "bluetooth speaker",
    "remote", "tv remote", "ac remote", "calculator", "digital watch",
    "circuit board", "pcb", "motherboard", "hard disk", "ram", "computer mouse", "keyboard",
    "cfl bulb", "tubelight", "led bulb", "fused bulb", "broken bulb",
    "banana peel", "fruit peel", "apple core", "orange peel", "mango peel", "mango seed",
    "vegetable peel", "potato skin", "onion peel", "garlic peel", "ginger peel",
    "rotten vegetable", "rotten tomato", "rotten potato", "spoiled food",
    "leftover food", "cooked rice", "pakhala", "curry", "dal", "bread crust", "roti",
    "egg shell", "broken egg shell", "boiled egg shell",
    "tea bag", "used tea bag", "tea leaves", "chai patti", "coffee grounds",
    "meat waste", "chicken bone", "fish bone", "prawn shell", "crab shell",
    "leaf plate", "khali", "sal leaf plate", "banana leaf", "food on leaf",
    "dry leaf", "dead leaves", "fallen leaves", "brown leaves",
    "twig", "stick", "branch", "dry grass", "hay", "straw",
    "flower waste", "dry flowers", "puja flowers", "garland", "marigold flowers",
    "coconut shell", "coconut husk", "coir", "tender coconut", "green coconut",
    "mask", "face mask", "surgical mask", "n95 mask", "used mask",
    "gloves", "latex gloves", "surgical gloves", "plastic gloves",
    "syringe", "injection", "needle", "medical waste",
    "cotton", "bloody cotton", "bandage", "bandaid", "gauze",
    "diaper", "baby diaper", "soiled diaper", "sanitary pad", "sanitary napkin",
    "razor", "shaving blade", "safety razor", "disposable razor",
    "paint can", "spray paint", "varnish tin", "chemical bottle", "pesticide bottle", "mosquito coil",
    "old cloth", "rag", "cleaning cloth", "torn clothes", "fabric scraps",
    "sock", "old socks", "torn sock", "shoe", "old shoe", "slipper", "broken chappal",
    "jute bag", "gunny bag", "rice sack",
    "pen", "plastic pen", "refill", "pencil", "eraser", "sharpener", "ruler",
    "rubber band", "hair clip", "comb", "broken comb", "toy", "plastic toy",
    "ceramic cup", "broken ceramic", "clay cup", "kulhad", "diya",
    "mixed waste", "garbage pile", "dust", "floor sweepings", "vacuum dust"
]

ADDITIONAL_LABELS = [
    "plastic hanger", "broken clothes hanger", "clothespin", "plastic peg",
    "plastic soap case", "toothbrush holder", "plastic loofah", "bath sponge",
    "plastic comb teeth", "hairbrush", "broken hairbrush", "plastic hairband", "hair clip",
    "plastic flower pot", "plant pot", "broken plastic pot", "artificial flower", "plastic leaves",
    "cable tie", "zip tie", "plastic rope", "nylon rope", "plastic twine",
    "plastic table mat", "placemat", "plastic tablecloth", "vinyl sheet",
    "raincoat", "torn raincoat", "plastic shower cap", "disposable shower cap",
    "plastic keychain", "id card holder", "laminate sheet", "plastic file folder",
    "plastic button", "sewing button", "plastic bead", "craft supplies",
    "cd cover", "dvd case", "jewel case", "plastic spool", "thread spool",
    "lego brick", "plastic block", "toy car", "plastic doll", "broken toy part",
    "plastic ball", "ping pong ball", "cricket ball plastic", "badminton shuttlecock",
    "balloon", "popped balloon", "rubber balloon", "water balloon",
    "plastic whistle", "party horn", "party hat", "plastic mask",
    "pen cap", "marker cap", "highlighter cap", "refill tube",
    "cardboard tube", "toilet roll core", "kitchen roll core", "paper core",
    "egg carton", "paper egg tray", "molded pulp tray",
    "file folder", "manila folder", "index card", "postcard", "greeting card", "visiting card",
    "paper confetti", "party popper waste", "crepe paper", "streamers",
    "wrapping paper", "gift wrap", "torn gift paper",
    "paperback book", "hardback book", "torn book cover", "old magazine page",
    "lottery ticket", "parking receipt", "toll receipt", "bank slip",
    "calendar page", "wall calendar", "desk calendar",
    "paper bag handle", "twisted paper handle", "paper mache",
    "rusty nail", "bent nail", "screw", "nut and bolt", "washer",
    "hinge", "door hinge", "metal latch", "padlock", "broken lock", "keys", "bunch of keys",
    "screwdriver", "wrench", "pliers", "hammer head", "saw blade",
    "metal pipe", "copper pipe", "iron pipe", "plumbing waste", "tap", "faucet",
    "metal ruler", "geometry compass", "metal clip", "binder clip",
    "aerosol can", "spray paint can", "deodorant can", "shaving foam can", "room freshener can",
    "metal bucket", "steel bucket", "metal paint bucket",
    "chain", "bicycle chain", "rusted chain", "metal spring",
    "zipper", "metal button", "snap button", "hook and eye",
    "perfume bottle", "cologne bottle", "scent bottle", "roll-on bottle",
    "nail polish bottle", "makeup foundation bottle", "cream jar",
    "broken mirror", "mirror shard", "hand mirror", "compact mirror",
    "picture frame glass", "window glass", "car window glass", "tempered glass",
    "glass marble", "glass bead", "crystal", "chandelier piece",
    "test tube", "laboratory glass", "beaker", "microscope slide",
    "glass bangle", "broken bangle", "glass jewelry",
    "jackfruit peel", "jackfruit skin", "kathal waste",
    "bitter gourd peel", "karela waste", "ridge gourd peel", "bottle gourd skin",
    "cauliflower stalk", "cauliflower stem", "cabbage core", "broccoli stem",
    "spinach root", "coriander root", "methi stalks", "curry leaf stem",
    "green chilli stem", "dried red chilli", "capsicum core", "bell pepper seeds",
    "drumstick peel", "drumstick fibre", "moringa waste",
    "tamarind shell", "imli seed", "tamarind fibre",
    "garlic skin", "ginger skin", "turmeric skin",
    "corn cob", "bhutta", "corn husk", "sweet corn waste",
    "mushroom stem", "radish leaves", "beetroot skin", "yam peel",
    "pomegranate peel", "pomegranate skin", "anaar chilka",
    "pineapple crown", "pineapple eye", "pineapple skin",
    "watermelon rind", "melon seeds", "muskmelon skin",
    "papaya seeds", "papaya skin",
    "custard apple skin", "sitaphal waste", "seeds",
    "grape stem", "rotten grapes", "dried raisins",
    "guava seeds", "rotten guava", "sapota skin", "chikoo peel",
    "litchi shell", "litchi seed", "jamun seed",
    "lemon seeds", "squeezed lemon", "dried lime",
    "fish scales", "fish head", "fish tail", "shrimp shell", "prawn head",
    "mutton fat", "mutton bone", "chicken skin", "meat trimmings",
    "crab claw", "lobster shell", "oyster shell", "mussel shell",
    "stale rice", "burnt rice", "spoiled curry", "sour dal",
    "fungus bread", "moldy roti", "stale cake", "melted ice cream",
    "used tea leaves", "coffee grounds", "wet tea bag",
    "panipuri leftovers", "soggy puri", "chutney waste",
    "pine cone", "acorn", "seed pod",
    "sawdust", "wood chips", "wood shavings", "sanding dust",
    "dried grass", "lawn clippings", "weeds", "uprooted weed",
    "neem leaves", "tulsi leaves", "banyan leaf", "peepal leaf",
    "stick", "twig", "bark", "tree root",
    "dead flower", "wilted bouquet", "dried garland",
    "hdmi cable", "vga cable", "ethernet cable", "lan wire",
    "audio jack", "aux cable", "rca cable",
    "power strip", "extension cord", "spike buster",
    "sim card", "micro sd card", "memory stick", "flash drive", "pen drive",
    "wifi dongle", "bluetooth dongle", "modem", "router",
    "webcam", "cctv camera", "broken camera lens",
    "mouse pad", "keyboard key", "laptop battery",
    "fitness band", "smart watch strap", "calculator screen",
    "leather belt", "broken belt", "leather wallet", "old purse",
    "leather shoe", "leather sandal", "shoe sole", "shoe insole",
    "backpack", "school bag", "torn bag", "duffel bag", "zipper pull",
    "canvas shoe", "running shoe", "sneaker", "shoe lace",
    "yoga mat", "foam mat", "rubber mat", "mouse mat",
    "rubber glove", "dishwashing glove", "latex glove",
    "elastic band", "waistband", "bra strap",
    "brick", "broken brick", "red brick piece",
    "concrete", "cement chunk", "plaster piece",
    "ceramic tile", "bathroom tile", "floor tile", "broken tile",
    "terracotta pot", "clay pot shard", "earthenware",
    "sand", "gravel", "pebble", "stone", "rock",
    "porcelain", "broken toilet seat", "sink fragment",
    "glass wool", "insulation foam", "drywall", "gypsum board",
    "cotton swab", "ear bud", "q-tip", "used cotton",
    "dental floss", "floss pick", "interdental brush",
    "razor blade", "shaving cartridge", "safety razor head",
    "wax strip", "hair removal strip",
    "makeup sponge", "beauty blender", "makeup wipe", "facial wipe",
    "sanitary pad wrapper", "tampon applicator",
    "condom wrapper", "pregnancy test kit",
    "soap sliver", "tiny soap piece", "shampoo bar",
    "cigarette ash", "cigar butt", "tobacco pouch",
    "chewing gum", "stuck gum", "gum wrapper",
    "candle wax", "melted candle", "birthday candle",
    "incense ash", "agarbatti stick", "matchstick", "burnt match",
    "chalk dust", "chalk piece", "blackboard eraser",
    "vacuum cleaner dust", "lint", "dryer lint", "floor sweepings"
]

STATES_PHYSICAL = ["crushed", "flattened", "broken", "shredded", "torn", "crumpled"]
STATES_DIRTY = ["dirty", "soiled", "muddy", "stained", "oily", "wet", "dry"]

SMART_BASE_ITEMS = [
    "plastic bottle", "water bottle", "milk packet", "chips packet", "biscuit wrapper",
    "plastic cup", "plastic container", "cardboard box", "pizza box", "paper cup",
    "newspaper", "tissue paper", "aluminum can", "soda can", "tin can", "food tin", "foil",
    "glass bottle", "glass jar", "mask", "glove", "battery", "wire", "cable",
    "plastic bag", "carry bag", "shampoo bottle", "medicine strip", "plate"
]

# --- GENERATE FULL LABEL LIST ---
FINAL_LABELS = []
FINAL_LABELS.extend(MANUAL_LABELS)
FINAL_LABELS.extend(ADDITIONAL_LABELS)

for item in SMART_BASE_ITEMS:
    for state in STATES_PHYSICAL:
        FINAL_LABELS.append(f"{state} {item}")
    for state in STATES_DIRTY:
        FINAL_LABELS.append(f"{state} {item}")

FINAL_LABELS = list(set(FINAL_LABELS))
FINAL_LABELS.sort()
LABELS = FINAL_LABELS

# ==========================================
# 2. MAPPINGS
# ==========================================
LABEL_TO_CATEGORY = {
    # PLASTICS (Added 'bag', 'cup', 'container')
    "pet": "plastic", "hdpe": "plastic", "ldpe": "plastic",
    "bottle": "plastic", "plastic": "plastic", "container": "plastic",
    "bag": "plastic", "cup": "plastic", "wrapper": "plastic", 
    "packet": "plastic", "pouch": "plastic", "polythene": "plastic",
    
    # PAPER / DRY
    "paper": "paper", "newspaper": "paper", "cardboard": "cardboard",
    "box": "cardboard", "carton": "cardboard", "book": "paper",
    "magazine": "paper", "envelope": "paper", "tissue": "paper",
    
    # METAL
    "can": "metal", "tin": "metal", "aluminum": "metal", "foil": "metal",
    "steel": "metal", "scrap": "metal",
    
    # GLASS
    "glass": "glass", "jar": "glass", "shard": "glass",
    
    # ORGANIC
    "peel": "organic", "core": "organic", "fruit": "organic",
    "vegetable": "organic", "food": "organic", "leftover": "organic",
    "egg": "organic", "shell": "organic", "leaf": "organic",
    "flower": "organic", "tea": "organic", "coffee": "organic",
    "rice": "organic", "curry": "organic", "bone": "organic",
    "meat": "organic", "rotti": "organic", "chapati": "organic",
    
    # E-WASTE
    "battery": "e-waste", "phone": "e-waste", "cable": "e-waste", 
    "wire": "e-waste", "circuit": "e-waste", "charger": "e-waste",
    "mouse": "e-waste", "keyboard": "e-waste",
    
    # HAZARDOUS
    "mask": "hazardous", "glove": "hazardous", "syringe": "hazardous",
    "bulb": "hazardous", "medicine": "hazardous"
}

def get_wet_dry(label):
    wet_keywords = ["peel", "core", "fruit", "vegetable", "food", "leftover", "egg", 
                    "tea", "coffee", "rice", "curry", "meat", "wet", "rotten"]
    if any(k in label for k in wet_keywords): return "WET / ORGANIC"
    return "DRY"

# --- LOAD MODELS ---
print("Loading Models...")
try:
    text_features = np.load("text_features.npy") 
    ort_session = ort.InferenceSession("visual_model.onnx", providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Error loading models: {e}")
    print("Ensure 'text_features.npy' and 'visual_model.onnx' are in the same folder.")
    sys.exit(1)

# --- USER INPUT ---
print("\n--- SETUP ---")
while True:
    b1 = input("Name Custom Bin 1 (e.g. plastic): ").strip().lower()
    b2 = input("Name Custom Bin 2 (e.g. organic): ").strip().lower()
    if b1 and b2 and b1 != b2: break
USER_BINS = ["recyclable", "mixed", b1, b2]
print(f"Bins: {USER_BINS}")

def preprocess(roi):
    img = cv2.resize(roi, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.481, 0.457, 0.408]) / [0.268, 0.261, 0.275]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0).astype(np.float32)

def get_bin(label, confidence):
    # Debug: Print logic
    print(f"\n[DEBUG] Analyzing '{label}' (Conf: {confidence:.2f})")
    
    # REMOVED CONFIDENCE CHECK: Always attempt to classify
    
    # 1. Custom Bin Keyword Match (Simple substring check)
    if USER_BINS[2] in label: 
        print(f" -> Keyword Match: '{USER_BINS[2]}' in label -> {USER_BINS[2]}")
        return USER_BINS[2]
    if USER_BINS[3] in label: 
        print(f" -> Keyword Match: '{USER_BINS[3]}' in label -> {USER_BINS[3]}")
        return USER_BINS[3]
    
    # 2. Category Match
    cat = "mixed"
    for key in LABEL_TO_CATEGORY:
        if key in label: 
            cat = LABEL_TO_CATEGORY[key]
            print(f" -> Found Category Keyword: '{key}' maps to '{cat}'")
            break
            
    if cat == USER_BINS[2]: 
        print(f" -> Category '{cat}' matches Bin 1 -> {USER_BINS[2]}")
        return USER_BINS[2]
    if cat == USER_BINS[3]: 
        print(f" -> Category '{cat}' matches Bin 2 -> {USER_BINS[3]}")
        return USER_BINS[3]
    
    # 3. Recyclable Check
    is_recycle = any(x in label for x in ["bottle", "can", "paper", "cardboard", "glass"])
    if is_recycle and "recyclable" in USER_BINS: 
        print(" -> Item identified as Recyclable -> recyclable")
        return "recyclable"
    
    print(" -> No rules matched -> Mixed")
    return "mixed"

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("\nSystem Ready!")
print("Press 'E' to classify the object in the green box.")
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    x1, y1 = (w - ROI_W)//2, (h - ROI_H)//2
    roi = frame[y1:y1+ROI_H, x1:x1+ROI_W]
    
    cv2.rectangle(frame, (x1, y1), (x1+ROI_W, y1+ROI_H), (0,255,0), 2)
    cv2.imshow("Smart Bin (Laptop)", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('e'):
        cv2.putText(frame, "Thinking...", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Smart Bin (Laptop)", frame)
        cv2.waitKey(1) 
        
        start = time.time()
        img_in = preprocess(roi)
        
        img_feats = ort_session.run(["image_features"], {"pixel_values": img_in})[0]
        img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True)
        
        logits = np.dot(img_feats, text_features.T)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probs = probs[0]
        
        idx = np.argsort(-probs)[0]
        try:
            label = LABELS[idx]
        except IndexError:
            label = f"Unknown (Index {idx})"
            
        conf = probs[idx]
        
        # Decision (Pass conf to get_bin now)
        final_bin = get_bin(label, conf)
        
        wet_dry_status = get_wet_dry(label)
        elapsed = time.time() - start
        
        print("-" * 30)
        print(f"Item: {label.upper()} ({conf:.2f})")
        print(f"Bin:  {final_bin.upper()}")
        print("-" * 30)
        
        res = frame.copy()
        cv2.putText(res, f"ITEM: {label.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(res, f"BIN: {final_bin.upper()}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        
        cv2.imshow("Smart Bin (Laptop)", res)
        cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()
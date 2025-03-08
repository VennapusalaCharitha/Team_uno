def toBraille(c):
    # Braille Unicode starts at 0x2800
    braille_start = 0x2800
    
    # Mapping of characters to Braille Unicode offsets
    mapping = {
        ' ': 0x0000,  # Space
        'a': 0x2801, 'b': 0x2803, 'c': 0x2809, 'd': 0x2819,
        'e': 0x2811, 'f': 0x280B, 'g': 0x281B, 'h': 0x2813,
        'i': 0x280A, 'j': 0x281A, 'k': 0x2805, 'l': 0x2807,
        'm': 0x280D, 'n': 0x281D, 'o': 0x2815, 'p': 0x280F,
        'q': 0x281F, 'r': 0x2817, 's': 0x280E, 't': 0x281E,
        'u': 0x2825, 'v': 0x2827, 'w': 0x283A, 'x': 0x282D,
        'y': 0x283D, 'z': 0x2835,
        '1': 0x2801, '2': 0x2803, '3': 0x2809, '4': 0x2819,
        '5': 0x2811, '6': 0x280B, '7': 0x281B, '8': 0x2813,
        '9': 0x280A, '0': 0x281A,
        ',': 0x2802, ';': 0x2806, ':': 0x2812, '.': 0x2832,
        '?': 0x2826, '!': 0x2816, "'": 0x2804, '"': 0x2808,
        '(': 0x2823, ')': 0x281C, '/': 0x280C, '\\': 0x2821,
        '-': 0x2824, '+': 0x2814, '*': 0x282C, '=': 0x283C,
        '<': 0x2822, '>': 0x2833, '@': 0x2820, '#': 0x2834,
        '$': 0x2829, '%': 0x282B, '^': 0x2831, '&': 0x2837,
        '_': 0x2830, '[': 0x282A, ']': 0x282E, '{': 0x282F,
        '}': 0x2836
    }
    
    # Convert character to lowercase for mapping
    c_lower = c.lower()
    
    # Get the Braille Unicode value from the mapping
    if c_lower in mapping:
        return chr(mapping[c_lower])
    else:
        return '?'  # Placeholder for unsupported characters

def converter(txt):
    tmp = ""
    for x in txt:
        tmp += toBraille(x)
    return tmp

# Get user input
txt = input("Please insert text: \n")
# Convert and print the result
print(converter(txt))
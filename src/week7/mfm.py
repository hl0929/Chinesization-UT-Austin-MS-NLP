

def forward_max_match(text, dictionary):
    max_len = max(len(word) for word in dictionary)
    i = 0
    tokens = []
    while i < len(text):
        match = None
        for j in range(max_len, 0, -1):
            if i + j <= len(text) and text[i:i+j] in dictionary:
                match = text[i:i+j]
                tokens.append(match)
                i += j
                break
        if not match:
            tokens.append(text[i])
            i += 1
    return tokens

dictionary = {"我", "爱", "北京", "天安门"}
text = "我爱北京天安门"
tokens = forward_max_match(text, dictionary)
print(tokens)
# Output: ['我', '爱', '北京', '天安门']
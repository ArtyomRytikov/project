# Пример функции для декодирования CTC
def ctc_decode(predictions, charset):
    # Greedy-декодирование
    text = ""
    last_char = -1
    for p in predictions:
        char_idx = np.argmax(p)
        if char_idx != last_char and char_idx < len(charset):
            text += charset[char_idx]
        last_char = char_idx
    return text
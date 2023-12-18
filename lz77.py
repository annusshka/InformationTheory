def find_longest_match(buffer, remaining):
    max_offset = 0
    max_length = 0

    for offset in range(1, len(buffer) + 1):
        for length in range(1, len(remaining) + 1):
            if remaining[:length] == buffer[-offset:][:length]:
                if length >= max_length:
                    max_offset = offset
                    max_length = length
            else:
                break

    return max_offset, max_length


def lz77_encode(message):
    encoded_message = []
    buffer = ""
    index = 0
    s = f"Message len = {len(message)}\n"

    while index < len(message):
        match = find_longest_match(buffer, message[index:])
        offset, length = match
        if offset != 0 and length != 0:
            next_char = message[index + length] if index + length < len(message) else ''
            encoded_message.append((offset, length, next_char))
            s += f"Буфер поиска: {buffer} | Буфер для предварительной записи сообщения: {message[index:]} |" + \
                 f" Кодовое слово: ({offset}, {length}, {next_char})\n"
            buffer += message[index:index + length + 1]
            index += length + 1
        else:
            encoded_message.append((0, 0, message[index]))
            s += f"Буфер поиска: {buffer} | Буфер для предварительной записи сообщения: {message[index:]} |" + \
                 f" Кодовое слово: (0, 0, {message[index]})\n"
            buffer += message[index]
            index += 1

    return s, encoded_message


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    s1 = "ПРОТОКОЛ ПРО ПРОТОКОЛ ПРОТОКОЛОМ ЗАПРОТОКОЛИРОВАЛИ"
    s1, res = lz77_encode(s1)
    write_ex("lz77_1", s1)

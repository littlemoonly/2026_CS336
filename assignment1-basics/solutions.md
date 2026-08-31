[toc]

### Problem (unicode1): Understanding Unicode (1 point)

##### (a) What Unicode character does chr(0) return?

`chr(0)` 返回的是 **空字符（null character）**，Unicode 码点(code point) 为 U+0000，也叫 NUL。

```
>>> chr(0)
'\x00'
```

##### (b) How does this character’s string representation (`__repr__()`) differ from its printed representation?

string representation / repr()：`'\x00'`（`repr()`会输出可见的调试形式）

printed：NUL是不可见的控制字符，因此什么也看不到

##### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

REPL 会用 `'\x00'` 表示 空字符

print 出来是不可见的，长度为1, i.e., `len(chr(0)) == 1`

如果将空字符写入文件，文件中就会有一个NUL控制字符（embeds an actual null byte (`0x00`)）

<img src="/Users/xinyue/Library/Application Support/typora-user-images/image-20260609155533439.png" alt="image-20260609155533439" style="zoom:60%;" />

### Problem (unicode2): Unicode Encodings (3 points)

##### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

> [!NOTE]
>
> 空间效率：
>
> | UTF-8                  | UTF-16                      | UTF-32                  |                           |
> | ---------------------- | --------------------------- | ----------------------- | ------------------------- |
> | 每个码点的字节数       | 1-4 字节（变长）            | 2 或 4 字节（变长）     | 固定 4 字节               |
> | ASCII 文本（英文）效率 | **最省**（每个字符 1 字节） | 浪费（每个字符 2 字节） | 最浪费（每个字符 4 字节） |
> | 字节顺序（BOM）        | 无歧义（字节流）            | 需要处理大端/小端       | 需要处理大端/小端         |
> | ASCII 兼容             | ✅ 向后兼容 ASCII            | ❌                       | ❌                         |

UTF-8 is the dominant encoding for the Internet, so it's compatible to real world data.

**空间效率**（UTF-8 对英文/ASCII 文本更紧凑）、**字节顺序无关**（UTF-8 没有 endianness 问题）、**向后兼容 ASCII**

##### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

<img src="/Users/xinyue/Library/Application Support/typora-user-images/image-20260609165132474.png" alt="image-20260609165132474" style="zoom:50%;" />

因为一个 UTF-8 Byte 不一定对应一个 Unicode Character，这个函数遇到多字节字符（如中文字符，每个 3 字节）就会把部分字节当成完整字符去解码，产生乱码。

例如 `>>> decode_utf8_bytes_to_str_wrong("你好呀".encode("utf-8"))`

![image-20260609165747974](/Users/xinyue/Library/Application Support/typora-user-images/image-20260609165747974.png)

##### (c) Give a two-byte sequence that does not decode to any Unicode character(s).

> [!NOTE]
>
> UTF-8 的多字节编码格式：
>
> | 字节数 | 第一个字节的二进制模式 | 后续字节的模式                   |
> | ------ | ---------------------- | -------------------------------- |
> | 1      | `0xxxxxxx`             | —                                |
> | 2      | `110xxxxx`             | `10xxxxxx`                       |
> | 3      | `1110xxxx`             | `10xxxxxx` `10xxxxxx`            |
> | 4      | `11110xxx`             | `10xxxxxx` `10xxxxxx` `10xxxxxx` |
>
> 如果一个字节以 `10xxxxxx` 开头（即字节值在 `0x80`-`0xBF` 范围，即 128-191），在 UTF-8 中它只能作为**后续字节（continuation byte）**，不能**单独出现**或**作为起始字节**。
>
> 另外，`0xC0` 和 `0xC1` 也是非法的（它们永远不会出现在合法的 UTF-8 中，因为它们是"过度编码"）。

```python
b'\x80\x80'.decode('utf-8')	# invalid start byte，continuation byte
b'\xc0\x80'.decode('utf-8')	#	invalid start byte，overlong encoding
```

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

##### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

最长单词如下

<img src="/Users/xinyue/Library/Application Support/typora-user-images/image-20260614105720045.png" alt="image-20260614105720045" style="zoom:70%;" />

##### (b) Profile your code. What part of the tokenizer training process takes the most time?

TODO
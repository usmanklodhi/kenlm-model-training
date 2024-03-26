import json
import os

class CharTokenizerConfig:
    def __init__(
            self,
            vocab: dict,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>"
        ):
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token


class CharTokenizer:
    def __init__(self, config_path=None):
        if config_path:
            self.__initialize_from_file(config_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "tokenizer_config_v2.json")
            if os.path.exists(default_path):
                self.__initialize_from_file(default_path)
            else:
                raise ValueError("Tokenizer config file was not provided. Attempted to load a default config file, but could not find one.")
                # self.__initialize_default()

    def __initialize_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config = CharTokenizerConfig(**config_data)
        self.__initialize(config)

    def __initialize(self, config: CharTokenizerConfig):
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token

        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        self.vocab = {}
        for combined_key, index in config.vocab.items():
            if combined_key not in self.special_tokens:
                hex_repr = combined_key.split(" ")[-1].replace("(", "").replace(")", "")
                char = chr(int(hex_repr, 16))
                self.vocab[char] = index
            else:
                self.vocab[combined_key] = index

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]

    def export_config(self):
        vocab = {}
        for char, index in self.vocab.items():
            if char not in self.special_tokens:
                hex_repr = "0x{:04x}".format(ord(char))
                combined_key = f"{char} ({hex_repr})"
                vocab[combined_key] = index
            else:
                vocab[char] = index

        return CharTokenizerConfig(vocab, *self.special_tokens)

    def export_config_file(self, file_path):
        config = self.export_config()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.__dict__, f, ensure_ascii=False, indent='\t')

    @classmethod
    def from_config_file(cls, file_path):
        instance = cls(config_path=file_path)
        return instance

    def tokenize(self, text, padding="max_length", max_length=128):
        # Add the BOS token
        token_ids = [self.bos_token_id]

        # Tokenize the text
        token_ids += [self.vocab.get(char, self.unk_token_id) for char in text]

        # Add the EOS token
        token_ids.append(self.eos_token_id)

        # Padding
        if len(token_ids) < max_length and padding == "max_length":
            token_ids += [self.pad_token_id] * (max_length - len(token_ids))
        elif len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        text = ""
        for _id in token_ids:
            _id = int(_id)
            token = self.reverse_vocab.get(_id, self.unk_token)
            if token == self.eos_token:
                break
            elif token in [self.pad_token, self.bos_token, self.unk_token] and skip_special_tokens:
                continue
            text += token
        return text
